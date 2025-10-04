# energy_model.py
# Units are arbitrary normalized units per-op
ENERGY_PER_DECIMAL_SYMBOL_OP = 1.0    # baseline for one decimal-digit op
ENERGY_PER_BINARY_BIT_OP = 0.3       # per single bit op cost
ECC_OVERHEAD_PER_WORD = 0.5          # cycles or energy per word for ECC (tune later)
ADC_CONVERSION_COST = 2.0            # conversion cost for analog <-> digital
