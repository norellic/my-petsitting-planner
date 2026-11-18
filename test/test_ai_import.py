"""Tests for ai_import: prompt building, provider calls, cleanup and applying.

No test touches the network - a fake transport stands in for the HTTP call.
"""

import json

import pytest

from ai_import import (
    AIImportError,
    ImportedTask,
    apply_import,
    normalize,
    parse_task_list,
)
from pawpal_system import Owner, Pet

RAW = [
    {"pet": "mochi", "title": "Feed breakfast", "duration_minutes": 10,
     "priority": "high", "repeats": "daily", "preferred_time": "Morning"},
    {"pet": "Mochi", "title": "Play", "duration_minutes": 5000,
     "priority": "urgent!!", "repeats": "sometimes", "preferred_time": "midnight"},
    {"pet": "Mochi", "title": "   "},  # no title: dropped
    "not a dict",
]


def gemini_reply(tasks):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"tasks": tasks})}]}}]}


def claude_reply(tasks):
    return {"content": [{"type": "tool_use", "name": "record_tasks", "input": {"tasks": tasks}}]}


def test_normalize_coerces_bad_values_and_matches_existing_pets():
    tasks = normalize(RAW, ["Mochi"], ["morning", "evening"])

    assert [t.title for t in tasks] == ["Feed breakfast", "Play"]
    first, second = tasks
    assert first.pet == "Mochi"  # matched ignoring case, not a new pet
    assert first.preferred_time == "morning"  # snapped to the real label
    assert (second.priority, second.repeats) == ("medium", "once")
    assert second.duration_minutes == 240  # clamped
    assert second.preferred_time is None  # unknown block ignored


def test_missing_pet_falls_back_only_when_there_is_exactly_one():
    raw = [{"title": "Walk", "duration_minutes": 20, "priority": "medium", "repeats": "once"}]

    assert normalize(raw, ["Mochi"], [])[0].pet == "Mochi"
    # Ambiguous: goes to the household rather than guessing a pet (or a pet "None").
    assert normalize(raw, ["Mochi", "Luna"], [])[0].pet == "Household"


def test_household_tasks_and_5_minute_durations():
    raw = [
        {"pet": "None", "title": "Mail", "duration_minutes": 3, "priority": "low", "repeats": "daily"},
        {"pet": "Household", "title": "Bucket", "duration_minutes": 12, "priority": "low", "repeats": "once"},
    ]

    mail, bucket = normalize(raw, ["Mochi", "Luna"], [])

    assert (mail.pet, bucket.pet) == ("Household", "Household")
    assert mail.species == "household"
    assert (mail.duration_minutes, bucket.duration_minutes) == (5, 10)  # step of 5, min 5


@pytest.mark.parametrize(
    "provider, reply",
    [("gemini", gemini_reply), ("claude", claude_reply)],
)
def test_each_provider_is_called_and_parsed(provider, reply):
    calls = []

    def transport(url, headers, body):
        calls.append((url, headers, body))
        return reply(RAW)

    tasks = parse_task_list("Feed Mochi", provider, "secret", ["Mochi"], ["morning"], transport)

    assert len(tasks) == 2
    url, headers, body = calls[0]
    assert "secret" in headers.values()
    assert "secret" not in url and "secret" not in json.dumps(body)  # key never in URL/body
    assert "Feed Mochi" in json.dumps(body)


def test_malformed_model_reply_is_a_friendly_error():
    with pytest.raises(AIImportError, match="expected format"):
        parse_task_list("x", "gemini", "k", [], [], lambda *_: {"candidates": []})
    with pytest.raises(AIImportError, match="expected format"):
        parse_task_list("x", "claude", "k", [], [], lambda *_: {"content": []})


@pytest.mark.parametrize(
    "notes, key, message",
    [("", "k", "notes first"), ("hi", "  ", "API key"), ("x" * 25_000, "k", "too long")],
)
def test_bad_input_is_rejected_before_any_call(notes, key, message):
    def boom(*_):
        raise AssertionError("should not call the network")

    with pytest.raises(AIImportError, match=message):
        parse_task_list(notes, "gemini", key, [], [], boom)


def test_no_tasks_found_is_an_error():
    with pytest.raises(AIImportError, match="No tasks"):
        parse_task_list("hello", "gemini", "k", [], [], lambda *_: gemini_reply([]))


def test_apply_import_creates_pets_and_reports_bad_rows():
    owner = Owner("Sam")
    owner.add_pet(Pet("Mochi", "dog"))
    rows = [
        ImportedTask(pet="mochi", title="Walk", duration_minutes=30, priority="high"),
        ImportedTask(pet="Luna", title="Feed", species="cat", repeats="daily"),
        ImportedTask(pet="Luna", title="Bad", duration_minutes=0),
    ]

    added, created, errors = apply_import(owner, rows)

    assert added == 2
    assert created == ["Luna"]
    assert len(errors) == 1 and "Bad" in errors[0]
    assert [p.name for p in owner.pets] == ["Mochi", "Luna"]
    assert owner.pets[1].species == "cat"
    assert len(owner.pets[0].tasks) == 1


def test_a_saved_parse_round_trips(tmp_path, monkeypatch):
    import ai_import

    monkeypatch.setattr(ai_import, "SAVE_DIR", tmp_path)
    tasks = [ImportedTask(pet="Wilson", title="Scoop", duration_minutes=5, priority="medium")]

    name = ai_import.save_parse("my test/../parse", "the notes", tasks)

    assert "/" not in name and ai_import.list_parses() == [name]
    assert ai_import.load_parse(name) == ("the notes", tasks)
    with pytest.raises(AIImportError):
        ai_import.load_parse("missing")


# --- agentic review ---------------------------------------------------------

from ai_import import ReviewResult, check_plan, diff_tasks, review_with_ai  # noqa: E402
from pawpal_system import TimeBlock  # noqa: E402

TWENTY_MIN = [TimeBlock("morning", "07:00", "07:20")]


def two_tasks():
    return [
        ImportedTask(pet="Wilson", title="Feed", duration_minutes=15, priority="high"),
        ImportedTask(pet="Wilson", title="Water", duration_minutes=15, priority="high"),
    ]


def as_raw(tasks):
    return [
        {"pet": t.pet, "title": t.title, "duration_minutes": t.duration_minutes,
         "priority": t.priority, "repeats": t.repeats, "preferred_time": ""}
        for t in tasks
    ]


def reviewer(revised, calls=None):
    def transport(url, headers, body):
        if calls is not None:
            calls.append(body)
        return gemini_reply_payload({"tasks": as_raw(revised), "summary": "Shortened water."})

    return transport


def gemini_reply_payload(payload):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def test_check_plan_reports_what_the_scheduler_dropped():
    check = check_plan(two_tasks(), TWENTY_MIN)

    assert check.scheduled == 1 and len(check.dropped) == 1
    assert check.high_dropped == 1
    assert (check.task_minutes, check.available_minutes) == (30, 20)


def test_review_accepts_a_revision_that_fits_better_and_uses_one_call():
    original = two_tasks()
    revised = two_tasks()
    revised[1].duration_minutes = 5  # water is a 5-minute chore
    calls = []

    result = review_with_ai("notes", original, "gemini", "k", TWENTY_MIN, reviewer(revised, calls))

    assert len(calls) == 1  # exactly one API call
    assert result.accepted and result.tasks[1].duration_minutes == 5
    assert result.before.dropped and not result.after.dropped
    assert result.changes == ["Water (Wilson): duration 15 min -> 5 min"]
    assert "Dropped" in json.dumps(calls[0])  # the scheduler result reached the model


def test_review_rejects_a_revision_that_deletes_tasks_to_make_things_fit():
    result = review_with_ai(
        "notes", two_tasks(), "gemini", "k", TWENTY_MIN, reviewer(two_tasks()[:1])
    )

    assert not result.accepted
    assert len(result.tasks) == 2  # original kept
    assert result.changes == []


def test_review_rejects_a_revision_that_fits_worse():
    worse = two_tasks()
    worse[0].duration_minutes = 20  # now Water can't fit even though 15 would have
    original = two_tasks()
    original[1].duration_minutes = 5  # this list fits fully

    result = review_with_ai("notes", original, "gemini", "k", TWENTY_MIN, reviewer(worse))

    assert not result.accepted
    assert result.tasks == original


def test_review_needs_availability():
    with pytest.raises(AIImportError, match="availability"):
        review_with_ai("notes", two_tasks(), "gemini", "k", [], reviewer([]))


def test_diff_tasks_reports_additions_and_removals():
    before = two_tasks()
    after = [before[0], ImportedTask(pet="Wilson", title="Close gates")]

    assert diff_tasks(before, after) == [
        "Added: Close gates (Wilson)",
        "Removed or reworded: Water (Wilson)",
    ]


# --- spacing ----------------------------------------------------------------------


def test_normalize_keeps_spacing_only_when_gap_and_group_are_both_present():
    raw = [
        {"pet": "George", "title": "Out 1", "min_gap_minutes": 178, "spread_group": "breaks"},
        {"pet": "George", "title": "Out 2", "min_gap_minutes": 180},  # no group
        {"pet": "George", "title": "Out 3", "spread_group": "breaks"},  # no gap
        {"pet": "George", "title": "Out 4", "min_gap_minutes": "lots", "spread_group": "breaks"},
        {"pet": "George", "title": "Out 5", "min_gap_minutes": 99999, "spread_group": "breaks"},
    ]

    tasks = normalize(raw, ["George"], [])

    assert [(t.min_gap_minutes, t.spread_group) for t in tasks] == [
        (180, "breaks"), (0, ""), (0, ""), (0, ""), (1440, "breaks"),
    ]


def test_spacing_reaches_the_real_tasks_and_the_reviewer():
    owner = Owner("Sam")
    rows = [
        ImportedTask(pet="George", title="Out", min_gap_minutes=180, spread_group="breaks"),
    ]
    apply_import(owner, rows)
    task = owner.pets[0].tasks[0]
    assert (task.min_gap_minutes, task.spread_group) == (180, "breaks")

    # Three breaks, two short windows: the check says the spacing is why one is dropped.
    breaks = [
        ImportedTask(pet="George", title=f"Out {n}", min_gap_minutes=180,
                     spread_group="breaks", preferred_time=pref)
        for n, pref in enumerate(["morning", None, "evening"], start=1)
    ]
    blocks = [TimeBlock("morning", "07:00", "09:00"), TimeBlock("evening", "17:30", "19:00")]
    check = check_plan(breaks, blocks)

    assert len(check.dropped) == 1
    assert "keeps 180 min" in check.dropped[0]
    calls = []

    def transport(url, headers, body):
        calls.append(json.dumps(body))
        return gemini_reply_payload({"tasks": as_raw(breaks), "summary": "Needs a midday window."})

    review_with_ai("notes", breaks, "gemini", "k", blocks, transport)

    assert "no slot keeps 180 min" in calls[0]  # the model is told why it was dropped
