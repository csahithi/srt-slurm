# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for dashboard components.

Tests timestamp parsing and other dashboard functionality.
"""

from datetime import datetime

import pytest


class TestRateMatchTab:
    """Tests for rate_match_tab module."""

    def test_parse_timestamp_yyyy_mm_dd(self):
        """Test parsing YYYY-MM-DD HH:MM:SS format."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts = "2025-12-30 15:52:38"
        dt = _parse_timestamp(ts)

        assert isinstance(dt, datetime)
        assert dt.year == 2025
        assert dt.month == 12
        assert dt.day == 30
        assert dt.hour == 15
        assert dt.minute == 52
        assert dt.second == 38

    def test_parse_timestamp_iso8601_with_microseconds(self):
        """Test parsing ISO 8601 format with microseconds and Z."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts = "2025-12-30T15:52:38.206058Z"
        dt = _parse_timestamp(ts)

        assert isinstance(dt, datetime)
        assert dt.year == 2025
        assert dt.month == 12
        assert dt.day == 30
        assert dt.hour == 15
        assert dt.minute == 52
        assert dt.second == 38

    def test_parse_timestamp_iso8601_without_z(self):
        """Test parsing ISO 8601 format without Z."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts = "2025-12-30T15:52:38.206058"
        dt = _parse_timestamp(ts)

        assert isinstance(dt, datetime)
        assert dt.year == 2025
        assert dt.month == 12
        assert dt.day == 30

    def test_parse_timestamp_iso8601_without_microseconds(self):
        """Test parsing ISO 8601 format without microseconds."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts = "2025-12-30T15:52:38"
        dt = _parse_timestamp(ts)

        assert isinstance(dt, datetime)
        assert dt.year == 2025
        assert dt.hour == 15

    def test_parse_timestamp_trtllm_format(self):
        """Test parsing TRTLLM MM/DD/YYYY-HH:MM:SS format."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts = "01/23/2026-08:04:38"
        dt = _parse_timestamp(ts)

        assert isinstance(dt, datetime)
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 23
        assert dt.hour == 8
        assert dt.minute == 4
        assert dt.second == 38

    def test_parse_timestamp_trtllm_various_dates(self):
        """Test parsing various TRTLLM timestamps."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        # End of year
        ts1 = "12/31/2025-23:59:59"
        dt1 = _parse_timestamp(ts1)
        assert dt1.year == 2025
        assert dt1.month == 12
        assert dt1.day == 31
        assert dt1.hour == 23

        # Start of year
        ts2 = "01/01/2026-00:00:00"
        dt2 = _parse_timestamp(ts2)
        assert dt2.year == 2026
        assert dt2.month == 1
        assert dt2.day == 1
        assert dt2.hour == 0

    def test_parse_timestamp_invalid(self):
        """Test that invalid timestamps raise ValueError."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        with pytest.raises(ValueError):
            _parse_timestamp("invalid-timestamp")

        with pytest.raises(ValueError):
            _parse_timestamp("not a date at all")

        with pytest.raises(ValueError):
            _parse_timestamp("2025-13-40 25:99:99")  # Invalid values

    def test_parse_timestamp_format_fallback(self):
        """Test that parser tries multiple formats in order."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        # Should parse successfully with any supported format
        formats = [
            ("2025-12-30 15:52:38", 2025),  # Standard
            ("2025-12-30T15:52:38.206058Z", 2025),  # ISO 8601 with Z
            ("01/23/2026-08:04:38", 2026),  # TRTLLM
        ]

        for ts, expected_year in formats:
            dt = _parse_timestamp(ts)
            assert dt.year == expected_year

    def test_parse_timestamp_time_delta(self):
        """Test that timestamps can be used for time delta calculations."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        ts1 = "01/23/2026-08:04:38"
        ts2 = "01/23/2026-08:04:40"

        dt1 = _parse_timestamp(ts1)
        dt2 = _parse_timestamp(ts2)

        delta = dt2 - dt1
        assert delta.total_seconds() == 2.0

    def test_parse_timestamp_mixed_formats(self):
        """Test parsing a sequence of different timestamp formats."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        # Simulate what dashboard might see from different backends
        timestamps = [
            "2025-12-30 15:52:38",  # Standard (could be from old cache)
            "2025-12-30T15:52:39.100000Z",  # SGLang
            "01/23/2026-08:04:40",  # TRTLLM
        ]

        dts = [_parse_timestamp(ts) for ts in timestamps]

        # All should parse successfully
        assert len(dts) == 3
        assert all(isinstance(dt, datetime) for dt in dts)

        # Should be able to compute deltas (even if not chronological)
        delta = dts[1] - dts[0]
        assert delta.total_seconds() == 1.1


class TestTimestampIntegration:
    """Integration tests for timestamp handling across parsers and dashboard."""

    def test_sglang_to_dashboard_pipeline(self):
        """Test that SGLang timestamps work through the entire pipeline."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp
        from analysis.srtlog.parsers import get_node_parser

        parser = get_node_parser("sglang")

        # SGLang format timestamp
        sglang_ts = "2025-12-30T15:52:38.206058Z"

        # Parser should be able to parse it
        dt_parser = parser.parse_timestamp(sglang_ts)

        # Dashboard should be able to parse it
        dt_dashboard = _parse_timestamp(sglang_ts)

        # Both should produce same datetime
        assert dt_parser.year == dt_dashboard.year
        assert dt_parser.month == dt_dashboard.month
        assert dt_parser.day == dt_dashboard.day
        assert dt_parser.hour == dt_dashboard.hour
        assert dt_parser.minute == dt_dashboard.minute
        assert dt_parser.second == dt_dashboard.second

    def test_trtllm_to_dashboard_pipeline(self):
        """Test that TRTLLM timestamps work through the entire pipeline."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp
        from analysis.srtlog.parsers import get_node_parser

        parser = get_node_parser("trtllm")

        # TRTLLM format timestamp
        trtllm_ts = "01/23/2026-08:04:38"

        # Parser should be able to parse it
        dt_parser = parser.parse_timestamp(trtllm_ts)

        # Dashboard should be able to parse it
        dt_dashboard = _parse_timestamp(trtllm_ts)

        # Both should produce same datetime
        assert dt_parser.year == dt_dashboard.year
        assert dt_parser.month == dt_dashboard.month
        assert dt_parser.day == dt_dashboard.day
        assert dt_parser.hour == dt_dashboard.hour
        assert dt_parser.minute == dt_dashboard.minute
        assert dt_parser.second == dt_dashboard.second

    def test_mixed_backend_timestamps_in_dashboard(self):
        """Test that dashboard can handle timestamps from mixed backends."""
        from analysis.dashboard.rate_match_tab import _parse_timestamp

        # Simulate dashboard receiving timestamps from different backends
        mixed_timestamps = [
            "2025-12-30T15:52:38.206058Z",  # SGLang
            "01/23/2026-08:04:38",  # TRTLLM
            "2025-12-30 15:52:38",  # Standard format
        ]

        # All should parse without error
        parsed = []
        for ts in mixed_timestamps:
            dt = _parse_timestamp(ts)
            parsed.append(dt)
            assert isinstance(dt, datetime)

        # Should be able to compute time deltas
        assert len(parsed) == 3
