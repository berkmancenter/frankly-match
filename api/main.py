from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import os
import random
import re
import time
import traceback
from uuid import uuid4
from typing import Literal, Optional

from logger import log, log_context
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from match import group_match
from text_match import (
    TextMatchingService,
    placeholder_responses,
    plan_group_sizes,
)


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


app = FastAPI(title="Frankly Match API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Match-Run-ID"],
)


def _adopt_request_context(request: Request) -> None:
    """Merge study/event linkage stashed on the request into the log context.

    The /match route is synchronous, so FastAPI runs it in a worker thread with
    a *copy* of the context: a ContextVar it sets never reaches the async
    exception handlers or this middleware's completion log. request.state is
    backed by the ASGI scope and is shared by all of them.
    """
    extra = getattr(request.state, "log_context", None)
    if extra:
        log_context.set({**log_context.get(), **extra})


@app.middleware("http")
async def match_run_context(request: Request, call_next):
    if request.url.path != "/match":
        return await call_next(request)
    run_id = str(uuid4())
    token = log_context.set({"match_run_id": run_id})
    started = time.monotonic()
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Starlette's outer error middleware would build the 500 itself,
            # without this header. Build it here so a failed attempt is still
            # identifiable, and keep the traceback: once the exception no
            # longer propagates, the server log would not print it either.
            _adopt_request_context(request)
            log.log_event("ERROR", "Match request failed", request, extra_data={
                "status_code": 500,
                "duration_seconds": time.monotonic() - started,
                "traceback": traceback.format_exc(),
            })
            response = JSONResponse(
                {"code": "INTERNAL_ERROR", "message": "Unexpected server error"},
                status_code=500,
            )
        else:
            _adopt_request_context(request)
            log.log_event("INFO", "Match request completed", request, extra_data={
                "status_code": response.status_code,
                "duration_seconds": time.monotonic() - started,
            })
        response.headers["X-Match-Run-ID"] = run_id
        return response
    finally:
        log_context.reset(token)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ParticipantData(BaseModel):
    binaryAnswerMask: str = ""
    freeTextResponse: Optional[str] = None
    # Identity is used only to link a participant to their pre-survey row, and
    # a person who cannot be linked still gets a table. So these are never
    # validated beyond trimming: a malformed email must not 422 the whole event.
    email: Optional[str] = None
    name: Optional[str] = None
    model_config = {"extra": "allow"}

    @field_validator("email", "name")
    @classmethod
    def blank_identity_is_absent(cls, v: Optional[str]) -> Optional[str]:
        return (v.strip() or None) if v is not None else None


class MatchRequest(BaseModel):
    algorithm: str
    targetGroupSize: int
    participants: dict[str, ParticipantData]
    studyId: Optional[str] = None
    eventId: Optional[str] = None

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
    """Client-facing group payload.

    assignedTarget, achievedDiversity and fallbackUsed are deliberately absent.
    They are matching diagnostics, not something the client renders, and they go
    to Google Cloud Logging as structured extra_data instead.
    """

    groupId: str
    participantIds: list[str]
    diversityLevel: Optional[Literal["high", "medium", "low", "unknown"]] = None
    diffusionStatement: Optional[str] = None


class MatchResponse(BaseModel):
    results: list[GroupResult]
    # participantResponses used to be returned here so a caller could see what
    # was actually embedded when a participant supplied no freeTextResponse.
    # That is the single largest thing in the payload on a big event, and it is
    # diagnostic rather than something the client acts on, so it is logged now.


class MissingTextResponses(Exception):
    def __init__(self, participant_ids: list[str]):
        self.participant_ids = participant_ids
        super().__init__(
            f"{len(participant_ids)} participant(s) have no freeTextResponse"
        )


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


def _error(
    code: str, message: str, status: int, request: Request | None = None
) -> JSONResponse:
    log.log_event("ERROR", f"Error {code}: {message}", request=request)
    return JSONResponse({"code": code, "message": message}, status_code=status)


@app.exception_handler(MissingTextResponses)
async def missing_text_handler(request: Request, exc: MissingTextResponses) -> JSONResponse:
    _adopt_request_context(request)
    log.log_event(
        "ERROR",
        f"Refusing to match: {len(exc.participant_ids)} participant(s) "
        f"missing freeTextResponse and REQUIRE_REAL_TEXT is set",
        request=request,
        extra_data={"participant_ids_missing_text": exc.participant_ids},
    )
    return JSONResponse(
        {
            "code": "MISSING_TEXT_RESPONSES",
            "message": "freeTextResponse missing or empty for: "
            + ", ".join(exc.participant_ids),
        },
        status_code=422,
    )


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
    return _error(code, msg, status, request)


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
    request: Request | None = None,
) -> dict[str, str]:
    placeholders = placeholder_responses(list(participants))
    responses: dict[str, str] = {}
    placeholder_ids: list[str] = []
    for participant_id, data in participants.items():
        supplied = data.freeTextResponse.strip() if data.freeTextResponse else ""
        # TODO: Remove placeholder responses once freeTextResponse is guaranteed in the payload.
        if not supplied:
            placeholder_ids.append(participant_id)
        responses[participant_id] = supplied or placeholders[participant_id]

    placeholder_set = set(placeholder_ids)
    # One-shot-event protection: groups formed from placeholder text are
    # meaningless but look statistically perfect, so a silent substitution at a
    # live event would burn the only demonstration. Strict mode turns the silent
    # fallback into a loud, listable refusal. Enable REQUIRE_REAL_TEXT=1 in the
    # deployed environment for real events; leave off for demos and tests.
    if placeholder_ids and os.getenv("REQUIRE_REAL_TEXT", "").lower() in {"1", "true", "yes"}:
        raise MissingTextResponses(placeholder_ids)

    log.log_event(
        "INFO",
        f"Collected {len(responses)} participant responses "
        f"({len(placeholder_ids)} placeholder)",
        request=request,
        extra_data={
            "participant_count": len(responses),
            "placeholder_count": len(placeholder_ids),
            "placeholder_participant_ids": placeholder_ids,
            # This is what used to ride back in MatchResponse.participantResponses.
            "responses": [
                {
                    "participant_id": participant_id,
                    "email": participants[participant_id].email,
                    "name": participants[participant_id].name,
                    "response": text,
                    "response_length": len(text),
                    "is_placeholder": participant_id in placeholder_set,
                }
                for participant_id, text in responses.items()
            ],
        },
    )
    return responses


def _group_size_report(
    expected_ids: Sequence[str],
    target_group_size: int,
    produced_groups: Sequence[Sequence[str]],
) -> dict:
    """Compare the groups actually produced against the planned sizing.

    Coverage is checked by identity rather than by count: equal counts can hide
    one participant duplicated and another dropped, which is exactly the failure
    the ERROR path claims to catch.
    """
    participant_count = len(expected_ids)
    group_sizes = [len(group) for group in produced_groups]
    assigned = [pid for group in produced_groups for pid in group]
    expected = set(expected_ids)
    seen = Counter(assigned)
    missing = sorted(expected - set(seen))
    duplicated = sorted(pid for pid, n in seen.items() if n > 1)
    unexpected = sorted(set(seen) - expected)

    report = {
        "participant_count": participant_count,
        "target_group_size": target_group_size,
        "group_count": len(group_sizes),
        "group_sizes": group_sizes,
        "size_histogram": {
            str(size): group_sizes.count(size) for size in sorted(set(group_sizes))
        },
        "participants_assigned": sum(group_sizes),
    }
    try:
        planned = plan_group_sizes(participant_count, target_group_size)
    except ValueError:
        planned = None
    if planned is not None:
        report["planned_group_count"] = len(planned)
        report["planned_group_sizes"] = planned
        report["matches_plan"] = sorted(planned) == sorted(group_sizes)
    report["missing_participants"] = missing
    report["duplicated_participants"] = duplicated
    report["unexpected_participants"] = unexpected
    report["all_participants_assigned"] = not (missing or duplicated or unexpected)
    return report


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
    # See _adopt_request_context: this thread's ContextVar is a private copy.
    request.state.log_context = {"study_id": req.studyId, "event_id": req.eventId}
    _adopt_request_context(request)
    if req.algorithm == "binaryGroupMatch":
        samples = _normalize_masks(req.participants)
        groups = group_match(samples, req.targetGroupSize)

        log.log_event(
            "INFO",
            f"Matched {len(req.participants)} participants into {len(groups)} "
            f"groups via binaryGroupMatch",
            request=request,
            extra_data={
                "algorithm": "binaryGroupMatch",
                "participants": samples,
                "groups": groups,
                "group_sizes": _group_size_report(
                    list(req.participants), req.targetGroupSize, groups
                ),
            },
        )
        return MatchResponse(
            results=[
                GroupResult(groupId=str(index + 1), participantIds=group)
                for index, group in enumerate(groups)
            ]
        )

    participant_responses = _text_responses(req.participants, request)
    groups = get_text_matching_service().match(
        participant_responses,
        req.targetGroupSize,
        request,
    )

    size_report = _group_size_report(
        list(req.participants),
        req.targetGroupSize,
        [group.participant_ids for group in groups],
    )
    # Keyed on diversity_level rather than assigned_target: targets are raw
    # distances on the embedding's scale, so counting by target value would put
    # every group in its own bucket. Levels are a fixed enum.
    condition_counts: dict[str, int] = {}
    for group in groups:
        level = group.diversity_level
        condition_counts[level] = condition_counts.get(level, 0) + 1

    log.log_event(
        "INFO",
        f"Matched {len(req.participants)} participants into {len(groups)} "
        f"groups via textGroupMatch",
        request=request,
        extra_data={
            "algorithm": "textGroupMatch",
            "group_sizes": size_report,
            "condition_counts": condition_counts,
            "fallback_group_count": sum(1 for group in groups if group.fallback_used),
            "groups": [
                {
                    "groupId": str(index + 1),
                    "participantIds": group.participant_ids,
                    "diversityLevel": group.diversity_level,
                    "assignedTarget": group.assigned_target,
                    "achievedDiversity": group.achieved_diversity,
                    "diffusionStatement": group.diffusion_statement,
                    "fallbackUsed": group.fallback_used,
                    "fallbackReason": (
                        "participant_embedding_failed" if group.diversity_level == "unknown"
                        else "statement_embedding_failed" if group.fallback_used else None
                    ),
                }
                for index, group in enumerate(groups)
            ],
        },
    )

    if not size_report.get("matches_plan", True):
        log.log_event(
            "WARNING",
            "Produced group sizes do not match the planned sizing",
            request=request,
            extra_data=size_report,
        )
    if not size_report["all_participants_assigned"]:
        log.log_event(
            "ERROR",
            f"{size_report['participants_assigned']} of "
            f"{size_report['participant_count']} participants were assigned to a group",
            request=request,
            extra_data=size_report,
        )
    if condition_counts.get("high", 0) < 2:
        log.log_event(
            "WARNING",
            f"Only {condition_counts.get('high', 0)} group(s) in the high "
            f"diversity arm; the contrast is not estimable at this event size",
            request=request,
            extra_data={"condition_counts": condition_counts},
        )

    return MatchResponse(
        results=[
            GroupResult(
                groupId=str(index + 1),
                participantIds=group.participant_ids,
                diversityLevel=group.diversity_level,
                diffusionStatement=group.diffusion_statement,
            )
            for index, group in enumerate(groups)
        ]
    )
