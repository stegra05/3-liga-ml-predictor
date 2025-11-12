#!/usr/bin/env python3
"""
3. Liga Match Predictor - Main Entry Point

This is the single entry point for all functionality. Running without arguments
defaults to predicting matches.

Usage:
    python main.py                    # Predict next matchday (default)
    python -m .                       # Same as above (package mode)
    python main.py predict            # Explicit prediction
    python main.py predict --help     # Show prediction options
    python main.py <subcommand>       # Run other commands
    python main.py <subcommand> --help # Show subcommand-specific help
    python main.py --help             # List all available commands
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def patched_argv(args):
    """Temporarily patch sys.argv for forwarding arguments to submodules"""
    old = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + list(args)
        yield
    finally:
        sys.argv = old


def import_module(module_path):
    """Import a module by its file path"""
    # Convert file path to module path
    # e.g., "scripts/collectors/fbref_collector.py" -> "scripts.collectors.fbref_collector"
    module_str = module_path.replace('/', '.').replace('.py', '')
    # Handle root-level modules
    if module_str.startswith('.'):
        module_str = module_str[1:]
    return __import__(module_str, fromlist=['main'])


def forward_to_module(module_path, args_list):
    """Forward arguments to a module's main() function"""
    mod = import_module(module_path)
    
    # Handle --help specially: show the module's help instead of forwarding
    if '--help' in args_list or '-h' in args_list:
        # Try to get help from the module's parser
        # Most modules create their parser in main(), so we need to call it
        # with --help to get the help output
        with patched_argv(['--help']):
            try:
                mod.main()
            except SystemExit as e:
                # argparse calls sys.exit(0) after printing help, which is expected
                # Re-raise if it's not exit code 0 (which indicates help was shown)
                if e.code != 0:
                    raise
        return
    
    with patched_argv(args_list):
        mod.main()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='3. Liga Match Predictor - Single entry point for all functionality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Predict next matchday
  python main.py predict --season 2025        # Predict specific season
  python main.py fetch-weather -- --limit 50  # Fetch weather data
  python main.py collect-fbref                # Collect FBref data
        """
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # ========================================================================
    # PREDICT SUBCOMMAND (default behavior)
    # ========================================================================
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict match results (default command)',
        description='Predict 3. Liga match results using Random Forest Classifier'
    )
    predict_parser.add_argument(
        '--season',
        type=str,
        help='Season to predict (e.g., 2025-2026). Default: current season'
    )
    predict_parser.add_argument(
        '--matchday',
        type=int,
        help='Specific matchday to predict. Default: next upcoming matchday'
    )
    predict_parser.add_argument(
        '--update-data',
        action='store_true',
        help='Update/fetch data for the matchday before predicting'
    )
    predict_parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force model retraining'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        help='Save predictions to CSV file'
    )
    predict_parser.add_argument(
        '--weather-mode',
        type=str,
        choices=['live', 'estimate', 'off'],
        default='live',
        help='Weather fetching mode: live (forecast API), estimate (historical), or off (defaults). Default: live'
    )
    predict_parser.add_argument(
        '--ext-data',
        action='store_true',
        help='Include heavy external data collection (FBref, matchday-level standings). Default: False'
    )
    
    # ========================================================================
    # COLLECTOR SUBCOMMANDS
    # ========================================================================
    collect_fbref_parser = subparsers.add_parser(
        'collect-fbref',
        help='Collect data from FBref',
        description='Collect team and player statistics from FBref'
    )
    collect_fbref_parser.add_argument('args', nargs=argparse.REMAINDER)
    collect_fbref_parser.set_defaults(
        _forward=('scripts/collectors/fbref_collector.py',)
    )
    
    collect_openligadb_parser = subparsers.add_parser(
        'collect-openligadb',
        help='Collect data from OpenLigaDB',
        description='Collect match data from OpenLigaDB API'
    )
    collect_openligadb_parser.add_argument('args', nargs=argparse.REMAINDER)
    collect_openligadb_parser.set_defaults(
        _forward=('scripts/collectors/openligadb_collector.py',)
    )
    
    collect_transfermarkt_parser = subparsers.add_parser(
        'collect-transfermarkt-referees',
        help='Collect referee data from Transfermarkt',
        description='Collect referee information from Transfermarkt'
    )
    collect_transfermarkt_parser.add_argument('args', nargs=argparse.REMAINDER)
    collect_transfermarkt_parser.set_defaults(
        _forward=('scripts/collectors/transfermarkt_referee_collector.py',)
    )

    collect_oddsportal_parser = subparsers.add_parser(
        'collect-oddsportal',
        help='Collect betting odds from OddsPortal',
        description='Collect 1X2 betting odds from OddsPortal.com'
    )
    collect_oddsportal_parser.add_argument('args', nargs=argparse.REMAINDER)
    collect_oddsportal_parser.set_defaults(
        _forward=('scripts/collectors/oddsportal_collector.py',)
    )

    # ========================================================================
    # PROCESSOR SUBCOMMANDS
    # ========================================================================
    export_ml_parser = subparsers.add_parser(
        'export-ml-data',
        help='Export ML training data',
        description='Export processed data for machine learning training'
    )
    export_ml_parser.add_argument('args', nargs=argparse.REMAINDER)
    export_ml_parser.set_defaults(
        _forward=('scripts/processors/ml_data_exporter.py',)
    )
    
    fetch_weather_parser = subparsers.add_parser(
        'fetch-weather',
        help='Fetch weather data for matches',
        description='Fetch historical weather data using Open-Meteo archive API'
    )
    fetch_weather_parser.add_argument('args', nargs=argparse.REMAINDER)
    fetch_weather_parser.set_defaults(
        _forward=('scripts/processors/fetch_weather.py',)
    )
    
    fetch_weather_dwd_parser = subparsers.add_parser(
        'fetch-weather-dwd',
        help='Fetch weather data from DWD',
        description='Fetch weather data from Deutscher Wetterdienst (DWD)'
    )
    fetch_weather_dwd_parser.add_argument('args', nargs=argparse.REMAINDER)
    fetch_weather_dwd_parser.set_defaults(
        _forward=('scripts/processors/fetch_weather_dwd.py',)
    )
    
    fetch_weather_meteostat_parser = subparsers.add_parser(
        'fetch-weather-meteostat',
        help='Fetch weather data from Meteostat',
        description='Fetch weather data from Meteostat API'
    )
    fetch_weather_meteostat_parser.add_argument('args', nargs=argparse.REMAINDER)
    fetch_weather_meteostat_parser.set_defaults(
        _forward=('scripts/processors/fetch_weather_meteostat.py',)
    )
    
    fetch_weather_multi_parser = subparsers.add_parser(
        'fetch-weather-multi',
        help='Fetch weather data from multiple sources',
        description='Fetch weather data using multi-source pipeline'
    )
    fetch_weather_multi_parser.add_argument('args', nargs=argparse.REMAINDER)
    fetch_weather_multi_parser.set_defaults(
        _forward=('scripts/processors/fetch_weather_multi.py',)
    )
    
    build_team_locations_parser = subparsers.add_parser(
        'build-team-locations',
        help='Build team location mappings',
        description='Build and update team location data'
    )
    build_team_locations_parser.add_argument('args', nargs=argparse.REMAINDER)
    build_team_locations_parser.set_defaults(
        _forward=('scripts/processors/build_team_locations.py',)
    )
    
    build_h2h_parser = subparsers.add_parser(
        'build-head-to-head',
        help='Build head-to-head statistics',
        description='Build head-to-head match statistics table'
    )
    build_h2h_parser.add_argument('args', nargs=argparse.REMAINDER)
    build_h2h_parser.set_defaults(
        _forward=('scripts/processors/build_head_to_head.py',)
    )
    
    import_data_parser = subparsers.add_parser(
        'import-existing-data',
        help='Import existing data files',
        description='Import data from existing CSV files into database'
    )
    import_data_parser.add_argument('args', nargs=argparse.REMAINDER)
    import_data_parser.set_defaults(
        _forward=('scripts/processors/import_existing_data.py',)
    )
    
    unify_teams_parser = subparsers.add_parser(
        'unify-teams',
        help='Unify duplicate team entries',
        description='Unify duplicate teams to canonical entries'
    )
    unify_teams_parser.add_argument('args', nargs=argparse.REMAINDER)
    unify_teams_parser.set_defaults(
        _forward=('scripts/processors/unify_teams.py',)
    )
    
    rating_calc_parser = subparsers.add_parser(
        'rating-calculator',
        help='Calculate team ratings',
        description='Calculate and update team rating statistics'
    )
    rating_calc_parser.add_argument('args', nargs=argparse.REMAINDER)
    rating_calc_parser.set_defaults(
        _forward=('scripts/processors/rating_calculator.py',)
    )
    
    # ========================================================================
    # UTILITY SUBCOMMANDS
    # ========================================================================
    team_mapper_parser = subparsers.add_parser(
        'team-mapper-init',
        help='Initialize team mappings',
        description='Initialize and update team name mappings'
    )
    team_mapper_parser.add_argument('args', nargs=argparse.REMAINDER)
    team_mapper_parser.set_defaults(
        _forward=('scripts/utils/team_mapper.py',)
    )
    
    weather_report_parser = subparsers.add_parser(
        'weather-coverage-report',
        help='Generate weather coverage report',
        description='Generate report on weather data coverage'
    )
    weather_report_parser.add_argument('args', nargs=argparse.REMAINDER)
    weather_report_parser.set_defaults(
        _forward=('scripts/utils/weather_coverage_report.py',)
    )
    
    # ========================================================================
    # DATABASE SUBCOMMANDS
    # ========================================================================
    db_init_parser = subparsers.add_parser(
        'db-init',
        help='Initialize database schema',
        description='Initialize database schema and tables'
    )
    db_init_parser.add_argument('args', nargs=argparse.REMAINDER)
    db_init_parser.set_defaults(
        _forward=('database/db_manager.py',)
    )
    
    # Derive known commands from subparsers instead of hardcoding
    # subparsers.choices is a dict mapping command names to their parsers
    # This is populated as we add parsers above, so it contains all commands
    known_commands = list(subparsers.choices.keys()) if hasattr(subparsers, 'choices') and subparsers.choices else []
    
    # Handle default behavior (no subcommand = predict)
    # Check if first argument (after script name) is a known command
    if len(sys.argv) == 1:
        # No arguments - default to predict
        from predict import main as predict_main
        predict_main()
        return
    elif len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        # First arg is not a command - treat as predict arguments
        # But handle --help specially
        if '--help' in sys.argv or '-h' in sys.argv:
            predict_parser.print_help()
            return
        # Forward all arguments to predict (it will parse them)
        from predict import main as predict_main
        predict_main()
        return
    
    # Parse arguments normally (we have a known subcommand), but keep unknowns
    # so we can forward them to target modules (e.g., collector flags).
    args, unknown = parser.parse_known_args()
    
    # Handle case where command is None (shouldn't happen, but safety check)
    if args.command is None:
        predict_parser.print_help()
        return
    
    # Handle forwarding subcommands
    if hasattr(args, '_forward'):
        module_path = args._forward[0]
        # Prefer unknown args captured by parse_known_args; fall back to 'args' remainder
        forward_args = unknown if 'unknown' in locals() and unknown else (args.args if hasattr(args, 'args') else [])
        # Strip a leading '--' sentinel if present (users sometimes add it for safety)
        if forward_args and forward_args[0] == '--':
            forward_args = forward_args[1:]
        forward_to_module(module_path, forward_args)
        return
    
    # Handle predict subcommand explicitly
    if args.command == 'predict':
        # Reconstruct arguments for predict.main()
        forwarded = []
        for opt in ('season', 'matchday', 'update_data', 'retrain', 'output', 'weather_mode', 'ext_data'):
            v = getattr(args, opt, None)
            if v is True:
                forwarded.append(f'--{opt.replace("_", "-")}')
            elif v not in (None, False):
                forwarded.extend([f'--{opt.replace("_", "-")}', str(v)])
        
        from predict import main as predict_main
        with patched_argv(forwarded):
            predict_main()
        return


if __name__ == '__main__':
    main()

