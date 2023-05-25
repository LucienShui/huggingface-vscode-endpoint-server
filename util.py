import logging
import argparse

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
    parser.add_argument('--pretrained', type=str, default='bigcode/starcoder')
    parser.add_argument('--bit_precission', type=int, default=16)
    parser.add_argument('--auth_prefix', type=str, default='<secret_key>')
    return parser
