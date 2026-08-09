### False positive profile

At the operating threshold (0.0248), the model raises 11,384 alerts on the test window, of which 9,015 are false alarms (79.2% of alerts). Against the 85,498 legitimate transactions in the window that is a base false-alarm rate of 10.54%.

**Legitimate populations flagged above base rate**

- Legitimate transactions where `card4` = `discover` are flagged at **2.69x base rate** (28.37% vs 10.54%), accounting for 240 false alarms across 846 transactions.
- Legitimate transactions where `ProductCD` = `C` are flagged at **2.28x base rate** (24.05% vs 10.54%), accounting for 1,952 false alarms across 8,118 transactions.
- Legitimate transactions where `amount_bucket` = `$500-1k` are flagged at **2.26x base rate** (23.86% vs 10.54%), accounting for 544 false alarms across 2,280 transactions.
- Legitimate transactions where `DeviceType` = `mobile` are flagged at **1.83x base rate** (19.28% vs 10.54%), accounting for 1,309 false alarms across 6,790 transactions.
- Legitimate transactions where `amount_bucket` = `$250-500` are flagged at **1.75x base rate** (18.49% vs 10.54%), accounting for 1,021 false alarms across 5,523 transactions.
- Legitimate transactions where `card6` = `credit` are flagged at **1.73x base rate** (18.20% vs 10.54%), accounting for 3,541 false alarms across 19,461 transactions.
- Legitimate transactions where `card4` = `__missing__` are flagged at **1.65x base rate** (17.42% vs 10.54%), accounting for 127 false alarms across 729 transactions.
- Legitimate transactions where `card6` = `__missing__` are flagged at **1.64x base rate** (17.31% vs 10.54%), accounting for 126 false alarms across 728 transactions.

**Recurring joint profiles**

- **Cluster 0** — 1,932 false alarms (21.4% of all false alarms). Distinguished by ProductCD=`C` (8.6x over-represented); DeviceType=`mobile` (5.3x over-represented); identity_present=`1` (4.1x over-represented); amount_bucket=`<$25` (5.9x over-represented). Median amount $29.76 against a legitimate median of $68.53.
- **Cluster 1** — 459 false alarms (5.1% of all false alarms). Distinguished by ProductCD=`W` (1.2x over-represented); card6=`credit` (2.2x over-represented); DeviceType=`__missing__` (1.2x over-represented); identity_present=`0` (1.2x over-represented); amount_bucket=`$500-1k` (20.5x over-represented). Median amount $994.00 against a legitimate median of $68.53.
- **Cluster 2** — 1,382 false alarms (15.3% of all false alarms). Distinguished by amount_bucket=`$100-250` (1.3x over-represented); hour_bucket=`00-06` (4.1x over-represented). Median amount $117.00 against a legitimate median of $68.53.
- **Cluster 3** — 4,065 false alarms (45.1% of all false alarms). Distinguished by amount_bucket=`$100-250` (1.5x over-represented); hour_bucket=`18-24` (1.4x over-represented). Median amount $117.00 against a legitimate median of $68.53.
- **Cluster 4** — 1,177 false alarms (13.1% of all false alarms). Distinguished by card_history=`>50_prior_txns` (1.2x over-represented). Median amount $92.00 against a legitimate median of $68.53.
