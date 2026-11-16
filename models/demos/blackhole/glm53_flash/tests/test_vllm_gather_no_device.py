"""The vLLM scheduler's gathering of new requests (`tools/glm53_vllm_rows.gather`: at most one new request a free
row): no device, no vLLM.

    pytest models/demos/blackhole/glm53_flash/tests/test_vllm_gather_no_device.py \
        --confcutdir=models/demos/blackhole/glm53_flash -p no:cacheprovider
"""

from __future__ import annotations

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows
from models.demos.blackhole.glm53_flash.tools.glm53_vllm_rows import gather

KW = dict(rows=None, decodes=20, partial=False, held_ms=None, batch=6, wait_ms=200.0)


def call(**over):
    return gather(**{**dict(waiting=3, free_seats=3), **KW, **over})


def test_off_and_nothing_to_gather():
    assert call(batch=0) == "plain"  # off: the plugin's own step
    assert call(batch=1) == "plain"
    assert call(waiting=0) == "plain"  # nothing waits
    assert call(partial=True) == "plain"  # a prefill under way (or a blocked one)
    assert call(decodes=0) == "plain"  # nothing would run meanwhile: prefill now
    assert call(free_seats=0) == "plain"  # no seat: nothing could be admitted, no hold's clock
    assert call(rows=0) == "plain"


def test_hold_until_enough_or_too_long():
    assert call() == "hold"  # 3 could be admitted, 6 wanted, no hold yet
    assert call(held_ms=150.0) == "hold"
    assert call(held_ms=200.0) == "release"  # held long enough: the prefill with what there is
    assert call(waiting=6, free_seats=6) == "release"  # 6 admissible at once
    assert call(waiting=9, free_seats=6) == "release"
    assert call(waiting=9, free_seats=5) == "hold"  # the seats bound what one prefill admits


def test_free_rows_bound_admission():
    # 8 free seats but only 4 rows with a free seat: one wave takes 4 (one a row), so the hold goes on
    assert call(waiting=8, free_seats=8, rows=4) == "hold"
    assert call(waiting=8, free_seats=8, rows=6) == "release"
    assert call(waiting=8, free_seats=8, rows=4, held_ms=250.0) == "release"


def test_rows_published():
    glm53_vllm_rows.reset()
    assert glm53_vllm_rows.free_rows() is None
    glm53_vllm_rows.publish(5)
    assert glm53_vllm_rows.free_rows() == 5
    glm53_vllm_rows.reset()
