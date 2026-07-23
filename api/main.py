import random
import re

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from match import group_match


app = FastAPI(title="Frankly Match API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ParticipantData(BaseModel):
    binaryAnswerMask: str = ""
    model_config = {"extra": "allow"}


class MatchRequest(BaseModel):
    algorithm: str
    targetGroupSize: int
    participants: dict[str, ParticipantData]

    @field_validator("algorithm")
    @classmethod
    def check_algorithm(cls, v: str) -> str:
        if v != "binaryGroupMatch":
            raise ValueError(f"Unknown algorithm '{v}'. Supported: binaryGroupMatch")
        return v

    @field_validator("targetGroupSize")
    @classmethod
    def check_group_size(cls, v: int) -> int:
        if v < 2:
            raise ValueError("targetGroupSize must be at least 2")
        return v

    @model_validator(mode="after")
    def validate_masks(self) -> "MatchRequest":
        if not self.participants:
            raise ValueError("participants map must contain at least one entry")
        for pid, data in self.participants.items():
            if not re.fullmatch(r"[01]*", data.binaryAnswerMask):
                raise ValueError(
                    f"participants.{pid}.binaryAnswerMask must contain only '0' and '1' characters"
                )
        return self


class GroupResult(BaseModel):
    groupId: str
    participantIds: list[str]


class MatchResponse(BaseModel):
    results: list[GroupResult]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_CODE_HINTS = {
    "binaryAnswerMask": "INVALID_ANSWER_MASK",
    "targetGroupSize": "TARGET_GROUP_SIZE_TOO_SMALL",
    "algorithm": "UNKNOWN_ALGORITHM",
    "participants": "NO_PARTICIPANTS",
}


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"code": code, "message": message}, status_code=status)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    msg = first["msg"].removeprefix("Value error, ")
    field = str(first.get("loc", [""])[-1])
    code = next((v for k, v in _CODE_HINTS.items() if k in msg or k in field), "INTERNAL_ERROR")
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


@app.get("/health", response_model=dict)
def health():
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    samples = _normalize_masks(req.participants)
    groups = group_match(samples, req.targetGroupSize)
    return MatchResponse(results=[
        GroupResult(groupId=str(i + 1), participantIds=g)
        for i, g in enumerate(groups)
    ])
