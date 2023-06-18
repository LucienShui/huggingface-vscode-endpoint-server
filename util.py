import argparse
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('app')

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--pretrained', type=str, default='starcoder')
    parser.add_argument('--api-type', type=str, default='code')
    parser.add_argument('--bit-precission', type=int, default=16)
    parser.add_argument('--auth-prefix', type=str, default='<secret_key>')
    parser.add_argument('--ssl-certificate', type=str)
    parser.add_argument('--ssl-keyfile', type=str)
    parser.add_argument('--dry-run', action='store_true')
    return parser
