/*───────────────────────────────────────────────────────────────────────────
  threshold_sweep.sas

  Fraud detection threshold sweep and cost-sensitive operating point selection.

  A SAS implementation of the same analysis as src/ftlab/threshold.py and
  src/ftlab/cost.py. Card fraud review operations in banking are frequently a
  SAS shop, and a threshold sweep is a natural PROC-shaped problem: sort once,
  accumulate once, tabulate.

  ── VERIFICATION STATUS ────────────────────────────────────────────────────
  This program is written against SAS 9.4 syntax and is NOT executed by this
  repository's CI, because no SAS licence is available on the build machine.
  Do not treat its output as verified.

  What IS verified: the algorithm below is transliterated to ANSI SQL in
  sql/threshold_sweep.sql, that SQL is executed against the same scored data in
  the test suite, and it reproduces the Python confusion counts and dollar
  totals exactly at the reported operating point. So the logic is cross-checked
  in a third implementation; only the SAS syntax itself is unexercised.

  To run: SAS OnDemand for Academics (free) or any SAS 9.4+ session.
      %let scored = /path/to/test_scores.csv;
      %include "threshold_sweep.sas";

  ── INPUT ──────────────────────────────────────────────────────────────────
  test_scores.csv, written by scripts/run_pipeline.py, one row per scored
  transaction in the held-out test window:
      TransactionID   integer
      TransactionDT   integer   seconds offset
      TransactionAmt  numeric   dollars
      isFraud         0/1       outcome
      score_raw       numeric   model score in [0,1]
      score_platt     numeric
      score_isotonic  numeric

  ── COST MODEL ─────────────────────────────────────────────────────────────
  A missed fraud costs the actual transaction amount. A false alarm costs
  analyst review labour plus a customer-friction proxy. These match the Python
  defaults so the two implementations are comparable.
 ───────────────────────────────────────────────────────────────────────────*/

%let scored          = ./test_scores.csv;   /* input path                    */
%let analyst_hourly  = 35;                  /* fully loaded $/hour           */
%let minutes_review  = 4;                   /* minutes per manual review     */
%let friction_fp     = 5;                   /* customer friction $ per FP    */
%let score_col       = score_raw;           /* which score to threshold on   */

/* Cost of one false positive: review labour + friction. */
%let cost_fp = %sysevalf(&analyst_hourly * &minutes_review / 60 + &friction_fp);

options nonotes;
title;

/*── 1. Load the scored transactions ───────────────────────────────────────*/
proc import datafile="&scored" out=work.scored dbms=csv replace;
    getnames=yes;
    guessingrows=max;
run;

/*── 2. Totals, needed as denominators throughout ──────────────────────────*/
proc sql noprint;
    select count(*), sum(isFraud), sum(isFraud * TransactionAmt)
      into :n_total   trimmed,
           :n_fraud   trimmed,
           :amt_fraud trimmed
      from work.scored;
quit;

%put NOTE: &n_total transactions, &n_fraud fraud, $&amt_fraud fraud value.;

/*── 3. Sort descending by score ───────────────────────────────────────────
  Every achievable operating point is "review the top k". Sorting once and
  accumulating gives the entire sweep in a single pass -- and at finer
  resolution than a fixed threshold grid, because every distinct score becomes
  an operating point.                                                        */
proc sort data=work.scored out=work.ranked;
    by descending &score_col;
run;

/*── 4. Cumulative sweep ───────────────────────────────────────────────────*/
data work.sweep;
    set work.ranked end=last;
    retain cum_tp 0 cum_tp_amt 0;

    n_alerts = _n_;                     /* reviewing the top _n_ by score    */
    threshold = &score_col;             /* the score at this cut point       */

    cum_tp     + isFraud;
    cum_tp_amt + (isFraud * TransactionAmt);

    tp = cum_tp;
    fp = n_alerts - tp;
    fn = &n_fraud - tp;
    tn = &n_total - n_alerts - fn;

    precision        = tp / n_alerts;
    recall           = tp / &n_fraud;
    false_alarm_rate = 1 - precision;
    if (precision + recall) > 0
        then f1 = 2 * precision * recall / (precision + recall);
        else f1 = 0;

    alert_rate   = n_alerts / &n_total;
    review_hours = n_alerts * (&minutes_review / 60);

    fraud_caught_dollars = cum_tp_amt;
    fraud_missed_dollars = &amt_fraud - cum_tp_amt;
    dollar_recall        = cum_tp_amt / &amt_fraud;

    /* Expected cost: missed fraud at its actual value, plus false alarms. */
    fp_cost    = fp * &cost_fp;
    fn_cost    = fraud_missed_dollars;
    total_cost = fp_cost + fn_cost;

    /* Ties: a threshold admits every transaction sharing that score, so only
       the last row of a tied block is a genuine operating point.            */
    if last or (&score_col ne lag(&score_col)) then valid_cut = 1;
    else valid_cut = 0;

    keep n_alerts threshold tp fp fn tn precision recall f1
         false_alarm_rate alert_rate review_hours
         fraud_caught_dollars fraud_missed_dollars dollar_recall
         fp_cost fn_cost total_cost valid_cut;
run;

/* lag() above compares against the PREVIOUS row, so shift the flag forward:
   a cut is valid where the NEXT score differs.                              */
data work.sweep;
    set work.sweep;
    next_thresh = ifn(_n_ < &n_total, 1, .);
    set work.sweep(firstobs=2 keep=threshold rename=(threshold=thresh_next))
        work.sweep(obs=1 keep=threshold rename=(threshold=thresh_next));
    valid_cut = (threshold ne thresh_next) or (_n_ = &n_total);
    drop next_thresh thresh_next;
run;

/*── 5. Cost-optimal operating point ───────────────────────────────────────*/
proc sql noprint;
    create table work.cost_optimal as
    select *
      from work.sweep
     where valid_cut = 1
       and total_cost = (select min(total_cost) from work.sweep where valid_cut = 1);
quit;

/*── 6. F1-optimal, for the comparison that motivates the whole exercise ───*/
proc sql noprint;
    create table work.f1_optimal as
    select *
      from work.sweep
     where valid_cut = 1
       and f1 = (select max(f1) from work.sweep where valid_cut = 1);
quit;

/*── 7. Operational table: cheapest way to reach each recall target ────────*/
data work.targets;
    do recall_target = 0.5, 0.6, 0.7, 0.8, 0.9;
        output;
    end;
run;

proc sql;
    create table work.operational as
    select t.recall_target,
           s.threshold        label="Threshold",
           s.n_alerts         label="Alerts",
           s.review_hours     label="Analyst hours",
           s.precision        label="Precision",
           s.recall           label="Recall",
           s.false_alarm_rate label="False alarm rate",
           s.fraud_caught_dollars label="Fraud $ caught",
           s.fraud_missed_dollars label="Fraud $ missed"
      from work.targets t
      join work.sweep   s
        on s.recall >= t.recall_target
       and s.valid_cut = 1
     group by t.recall_target
    having s.n_alerts = min(s.n_alerts)   /* cheapest row meeting the target */
     order by t.recall_target;
quit;

/*── 8. Cost sensitivity: how far does the optimum move? ───────────────────
  The friction term is an assumption nobody can measure, so the stability of
  the recommendation matters more than the recommendation.                   */
data work.fp_costs;
    do cost_per_fp = 1, 2.5, 5, 7.3333, 10, 15, 25, 50, 100;
        output;
    end;
run;

proc sql;
    create table work.sensitivity as
    select c.cost_per_fp,
           s.threshold,
           s.n_alerts,
           s.review_hours,
           s.recall,
           s.precision,
           (s.fraud_missed_dollars + s.fp * c.cost_per_fp) as total_cost
      from work.fp_costs c
      join work.sweep    s on s.valid_cut = 1
     group by c.cost_per_fp
    having calculated total_cost = min(calculated total_cost)
     order by c.cost_per_fp;
quit;

/*── 9. Report ─────────────────────────────────────────────────────────────*/
options notes;

title "Cost-optimal operating point (FP cost = $&cost_fp)";
proc print data=work.cost_optimal noobs label;
    var threshold n_alerts review_hours precision recall
        fraud_caught_dollars fraud_missed_dollars total_cost;
    format threshold 8.4 review_hours 8.1 precision recall 8.4
           fraud_caught_dollars fraud_missed_dollars total_cost dollar14.2;
run;

title "F1-optimal operating point, for comparison";
proc print data=work.f1_optimal noobs label;
    var threshold n_alerts review_hours precision recall f1 total_cost;
    format threshold 8.4 review_hours 8.1 precision recall f1 8.4
           total_cost dollar14.2;
run;

title "Operational table: what each recall target costs";
proc print data=work.operational noobs label;
    format threshold 8.4 review_hours 8.1 precision recall false_alarm_rate 8.4
           fraud_caught_dollars fraud_missed_dollars dollar14.2;
run;

title "Sensitivity of the optimum to the false-positive cost assumption";
proc print data=work.sensitivity noobs label;
    format cost_per_fp dollar8.2 threshold 8.4 review_hours 8.1
           recall precision 8.4 total_cost dollar14.2;
run;

title;
