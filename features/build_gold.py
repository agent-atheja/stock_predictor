"""Entry point: build the Gold feature+label table.  python -m features.build_gold"""
from features.assembler import build_gold

if __name__ == "__main__":
    build_gold()
