"""Data collection modules"""

from .openligadb import OpenLigaDBCollector
from .fbref import FBrefCollector
from .fotmob import FotMobCollector
from .oddsportal import OddsPortalCollector
from .transfermarkt import TransfermarktRefereeCollector

__all__ = [
    'OpenLigaDBCollector',
    'FBrefCollector',
    'FotMobCollector',
    'OddsPortalCollector',
    'TransfermarktRefereeCollector'
]
