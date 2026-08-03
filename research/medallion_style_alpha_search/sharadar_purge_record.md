# Sharadar raw-data purge record — 2026-07-15

Per Nasdaq Data Link Personal Use terms §5.3 (purge within 30 days of
termination), the raw Sharadar export was deleted from this machine:

- `_data\sharadar\SHARADAR_SEP_2_da2386a176421f8ccbec6fabe5d11c0e.csv`
- `_data\sharadar\SHARADAR_SF1_3_a158fbb8637a13efbab2ba75fc06dc74.csv`
- `_data\sharadar\samples\SHARADAR_SEP_sample.csv`
- `_data\sharadar\samples\SHARADAR_SF1_sample.csv`
- `_data\sharadar\dev\sep_dev.parquet`
- `_data\sharadar\dev\sf1_dev.parquet`
- `_data\sharadar\confirm\sep_full.parquet`
- `_data\sharadar\confirm\sf1_full.parquet`
- `_data\sharadar\dev1_top1000.log`
- `_data\sharadar\dev1b_top1000_cap.log`
- `_data\sharadar\dev2a_top500_cap.log`
- `_data\sharadar\dev2b_top1500_cap.log`
- `_data\sharadar\dev2c_dv5m_cap.log`
- `_data\sharadar\dev3_top1000_cap_20bps.log`
- `_data\sharadar\dev_baseline.log`
- `_data\sharadar\probe.log`
- `_data\sharadar\study_run.log`
- `_data\sharadar\study_run2.log`
- `_data\sharadar\study_run.err`
- `_data\sharadar\probe.bat`
- `_data\sharadar\run_dev_baseline.bat`
- `_data\sharadar\run_dev_ladder.bat`
- `_data\sharadar\run_study.bat`
- `_data\sharadar\dev_ladder_done.marker`
- `_data\sharadar\ndl_api_key.txt`

Files remaining under `_data/sharadar/` after purge: 0

Derived Data retained under §6.2 (owned outright): the banked study verdict,
validation statistics and learned weights in research/medallion_style_alpha_search/.

---

## ADDENDUM — 2026-07-27: data re-exported, subscription still live

The purge above was executed on 2026-07-15, **before** the paid term ended. The
operator confirmed on 2026-07-27 that the SFA subscription is still inside its paid
month, and all seven tables were verified accessible (7/7, HTTP 200) before any
download began. Re-exporting licensed Data during an active subscription is ordinary
use, not a licence breach; the §5.3 obligation attaches to termination, which has not
yet taken effect.

**The statement above is therefore no longer current and must not be read as the
present state of this machine.** It is retained unedited as the record of what happened
on 2026-07-15.

Re-exported 2026-07-25..27 via `scripts/download_sharadar_data.py` into
`_data/sharadar/` (gitignored), with SHA-256, row counts and vendor snapshot times
recorded in `_data/sharadar/download_manifest.json`:

| Table | Rows | Size |
|---|---|---|
| SEP | 46,235,528 | 3,222 MB |
| SF3 | 79,190,744 | 2,892 MB |
| DAILY | 39,973,270 | 2,487 MB |
| SF1 | 3,200,111 | 2,401 MB |
| SF2 | 11,822,993 | 1,216 MB |
| ACTIONS | 671,240 | 47 MB |
| TICKERS | 78,883 | 26 MB |
| **total** | **181,172,769** | **12.3 GB** |

The prior programme held SF1 + SEP only. SF2, SF3, DAILY, ACTIONS and TICKERS have
never been read by any code in this repository.

**Purge obligation, restated:** when the subscription genuinely terminates, re-run
`python scripts/purge_sharadar_data.py --confirm` within 30 days. That script's globs
already cover every file listed above (`*.csv`, `*.zip`, `*.parquet`, `ndl_api_key.txt`)
and it will write a fresh dated record. **Outstanding:** the manifest
(`download_manifest.json`) is not matched by the current purge globs — it contains no
Data, only hashes and row counts, but the glob list should be reviewed at purge time.

---

## Purge — 2026-08-03

Per Nasdaq Data Link Personal Use terms §5.3 (purge within 30 days of
termination), the raw Sharadar export was deleted from this machine:

- `_data\sharadar\ACTIONS.csv`
- `_data\sharadar\DAILY.csv`
- `_data\sharadar\SEP.csv`
- `_data\sharadar\SF1.csv`
- `_data\sharadar\SF2.csv`
- `_data\sharadar\SF3.csv`
- `_data\sharadar\TICKERS.csv`
- `_data\sharadar\panel\dailyvol_dev.parquet`
- `_data\sharadar\panel\delistings.parquet`
- `_data\sharadar\panel\features_dev.parquet`
- `_data\sharadar\panel\insider_clustering_monthly.parquet`
- `_data\sharadar\panel\monthly_panel_dev.parquet`
- `_data\sharadar\panel\prices_to_2015-12-31.parquet`
- `_data\sharadar\panel\quality_art_dev.parquet`
- `_data\sharadar\panel\quality_art_dev_lowvol_retest.parquet`
- `_data\sharadar\panel\quarter_end_marketcap_dev.parquet`
- `_data\sharadar\panel\risk_features_dev.parquet`
- `_data\sharadar\panel\sf1_arq_raw.parquet`
- `_data\sharadar\panel\sf1_shares_dev.parquet`
- `_data\sharadar\panel\sf2_dev_purchases.parquet`
- `_data\sharadar\panel\sf2_dev_raw.parquet`
- `_data\sharadar\panel\sf3_ownership_dev.parquet`
- `_data\sharadar\download_manifest.json`
- `_data\sharadar\ndl_api_key.txt`

Files remaining under `_data/sharadar/` after purge: 0

Derived Data retained under §6.2 (owned outright): the banked study verdict,
validation statistics and learned weights in research/medallion_style_alpha_search/.

Provenance manifest rescued to `research\medallion_style_alpha_search\sharadar_download_manifest.json` (hashes and row counts only, no Data).
