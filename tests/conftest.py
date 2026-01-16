"""Shared pytest fixtures for gtape_prologix_drivers tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_serial():
    """Create a mock serial port."""
    mock = MagicMock()
    mock.is_open = True
    mock.in_waiting = 0
    mock.timeout = 6.0
    return mock


@pytest.fixture
def adapter(mock_serial):
    """Create adapter with mocked serial port."""
    with patch('gtape_prologix_drivers.adapter.serial.Serial', return_value=mock_serial):
        from gtape_prologix_drivers.adapter import PrologixAdapter
        adapter = PrologixAdapter('COM1', 10)
        yield adapter


@pytest.fixture
def mock_adapter():
    """Create a generic mock adapter for instrument tests."""
    mock = MagicMock()
    mock.ser = MagicMock()
    mock.ser.in_waiting = 0
    return mock


@pytest.fixture
def psu(mock_adapter):
    """Create E3631A PSU with mock adapter."""
    from gtape_prologix_drivers.instruments.agilent_e3631a import AgilentE3631A
    return AgilentE3631A(mock_adapter)


@pytest.fixture
def dmm(mock_adapter):
    """Create HP34401A DMM with mock adapter."""
    from gtape_prologix_drivers.instruments.hp34401a import HP34401A
    return HP34401A(mock_adapter)


@pytest.fixture
def scope(mock_adapter):
    """Create TDS460A scope with mock adapter."""
    from gtape_prologix_drivers.instruments.tds460a import TDS460A
    return TDS460A(mock_adapter)


@pytest.fixture
def load(mock_adapter):
    """Create PLZ164W load with mock adapter."""
    from gtape_prologix_drivers.instruments.plz164w import PLZ164W
    return PLZ164W(mock_adapter)
