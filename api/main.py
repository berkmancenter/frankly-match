from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
import re
from typing import Literal, Optional

from logger import log
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from match import group_match
from text_match import TextMatchingService, placeholder_responses


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


app = FastAPI(title="Frankly Match API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ParticipantData(BaseModel):
    binaryAnswerMask: str = ""
    freeTextResponse: Optional[str] = None
    model_config = {"extra": "allow"}


class MatchRequest(BaseModel):
    algorithm: str
    targetGroupSize: int
    participants: dict[str, ParticipantData]

    @field_validator("algorithm")
    @classmethod
    def check_algorithm(cls, v: str) -> str:
        supported = {"binaryGroupMatch", "textGroupMatch"}
        if v not in supported:
            raise ValueError(
                f"Unknown algorithm '{v}'. Supported: binaryGroupMatch, textGroupMatch"
            )
        return v

    @field_validator("targetGroupSize")
    @classmethod
    def check_group_size(cls, v: int) -> int:
        if v < 2:
            raise ValueError("targetGroupSize must be at least 2")
        return v

    @model_validator(mode="after")
    def validate_algorithm_inputs(self) -> "MatchRequest":
        if not self.participants:
            raise ValueError("participants map must contain at least one entry")

        if self.algorithm == "binaryGroupMatch":
            for pid, data in self.participants.items():
                if not re.fullmatch(r"[01]*", data.binaryAnswerMask):
                    raise ValueError(
                        f"participants.{pid}.binaryAnswerMask must contain only '0' and '1' characters"
                    )
        else:
            if self.targetGroupSize < 3:
                raise ValueError(
                    "textGroupMatch targetGroupSize must be at least 3"
                )
            if len(self.participants) < 3:
                raise ValueError(
                    "textGroupMatch requires at least 3 participants"
                )
        return self


class GroupResult(BaseModel):
    groupId: str
    participantIds: list[str]
    diversityLevel: Optional[Literal["high", "medium", "low", "unknown"]] = None
    assignedTarget: Optional[float] = None
    achievedDiversity: Optional[float] = None
    normalizedAchievedDiversity: Optional[float] = None
    diffusionStatement: Optional[str] = None
    fallbackUsed: Optional[bool] = None


class MatchResponse(BaseModel):
    results: list[GroupResult]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_CODE_HINTS = {
    "at least 3 participants": "INSUFFICIENT_PARTICIPANTS",
    "freeTextResponse": "INVALID_TEXT_RESPONSE",
    "binaryAnswerMask": "INVALID_ANSWER_MASK",
    "targetGroupSize": "TARGET_GROUP_SIZE_TOO_SMALL",
    "algorithm": "UNKNOWN_ALGORITHM",
    "participants": "NO_PARTICIPANTS",
}


def _error(code: str, message: str, status: int) -> JSONResponse:
    log.log_event("ERROR", f"Error {code}: {message}", request=None)
    return JSONResponse({"code": code, "message": message}, status_code=status)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    first = exc.errors()[0]
    msg = first["msg"].removeprefix("Value error, ")
    field = str(first.get("loc", [""])[-1])
    code = next(
        (value for hint, value in _CODE_HINTS.items() if hint in msg or hint in field),
        "INTERNAL_ERROR",
    )
    status = 400 if code in {"TARGET_GROUP_SIZE_TOO_SMALL", "UNKNOWN_ALGORITHM"} else 422
    return _error(code, msg, status)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _normalize_masks(participants: dict[str, ParticipantData]) -> dict[str, str]:
    """Random-pad shorter masks to the longest length in the request."""
    masks = {pid: p.binaryAnswerMask for pid, p in participants.items()}
    max_len = max((len(m) for m in masks.values()), default=0)
    if max_len == 0:
        return masks
    return {
        pid: mask + "".join(random.choice("01") for _ in range(max_len - len(mask)))
        for pid, mask in masks.items()
    }


def _text_responses(
    participants: dict[str, ParticipantData],
) -> dict[str, str]:
    placeholders = placeholder_responses(list(participants))
    responses = {}
    for participant_id, data in participants.items():
        # TODO: Remove placeholder responses once freeTextResponse is guaranteed in the payload.
        responses[participant_id] = (
            data.freeTextResponse.strip()
            if data.freeTextResponse and data.freeTextResponse.strip()
            else placeholders[participant_id]
        )
        log.log_event(
            "INFO",
            "Received participant response",
            request=None,
            extra_data={"participant_id": participant_id, "response_length": len(responses[participant_id])},
        )
    return responses


@lru_cache(maxsize=1)
def get_text_matching_service() -> TextMatchingService:
    return TextMatchingService.from_environment()


@app.get("/health", response_model=dict)
def health():
    return {"status": "ok"}


@app.post(
    "/match",
    response_model=MatchResponse,
    response_model_exclude_none=True,
)
def match(req: MatchRequest, request: Request):
    if req.algorithm == "binaryGroupMatch":
        samples = _normalize_masks(req.participants)
        groups = group_match(samples, req.targetGroupSize)

        log.log_event("INFO", f"Matched {len(req.participants)} participants into {len(groups)} groups via binaryGroupMatch", request=request)
        return MatchResponse(
            results=[
                GroupResult(groupId=str(index + 1), participantIds=group)
                for index, group in enumerate(groups)
            ]
        )

    participant_responses = _text_responses(req.participants)
    groups = get_text_matching_service().match(
        participant_responses,
        req.targetGroupSize,
    )
    log.log_event("INFO", f"Matched {len(req.participants)} participants into {len(groups)} groups via textMatchingService", request=request)
    return MatchResponse(
        results=[
            GroupResult(
                groupId=str(index + 1),
                participantIds=group.participant_ids,
                diversityLevel=group.diversity_level,
                assignedTarget=group.assigned_target,
                achievedDiversity=group.achieved_diversity,
                normalizedAchievedDiversity=group.normalized_achieved_diversity,
                diffusionStatement=group.diffusion_statement,
                fallbackUsed=group.fallback_used,
            )
            for index, group in enumerate(groups)
        ]
    )
