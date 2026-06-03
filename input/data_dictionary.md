# Data Dictionary

Source: 1991 Survey of Income and Program Participation (SIPP), in the 401(k)
application distributed through DoubleML and used in Chernozhukov et al. (2018)
and Abadie (2003).

| Variable | Type | Description | Units / Values |
|----------|------|-------------|----------------|
| `net_tfa` | Continuous | Net total financial assets | USD |
| `p401` | Binary | Participates in a 401(k) plan | 0 / 1 |
| `e401` | Binary | Eligible for a 401(k) plan | 0 / 1 |
| `age` | Continuous | Age of household head | Years |
| `inc` | Continuous | Household income | USD |
| `educ` | Continuous | Years of education | Years |
| `fsize` | Discrete | Family size | Count |
| `marr` | Binary | Married household indicator | 0 / 1 |
| `twoearn` | Binary | Two-earner household indicator | 0 / 1 |
| `db` | Binary | Defined-benefit pension indicator | 0 / 1 |
| `pira` | Binary | IRA participation indicator | 0 / 1 |
| `hown` | Binary | Home ownership indicator | 0 / 1 |

Variables used in the Table 3 replication:

- Outcome: `net_tfa`
- Treatment: `p401`
- Instrument: `e401`
- Controls: `age`, `inc`, `educ`, `fsize`, `marr`, `twoearn`, `db`, `pira`, `hown`
