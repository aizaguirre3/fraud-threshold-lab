-- ---------------------------------------------------------------------------
-- threshold_sweep.sql
--
-- The threshold sweep and cost-sensitive operating point, in ANSI SQL with
-- window functions. This is the same algorithm as sas/threshold_sweep.sas and
-- src/ftlab/threshold.py, written a third time.
--
-- Unlike the SAS program, this one IS executed -- the test suite runs it
-- against the scored test window and asserts it reproduces the Python
-- confusion counts and dollar totals exactly. That is what makes the shared
-- algorithm verified rather than merely written down.
--
-- Parameters are substituted by the caller:
--   {scored}   path to test_scores.csv
--   {cost_fp}  dollar cost of one false positive
-- ---------------------------------------------------------------------------

WITH scored AS (
    SELECT
        "isFraud"        AS is_fraud,
        "TransactionAmt" AS amount,
        {score_col}      AS score
    FROM read_csv_auto('{scored}')
),

totals AS (
    SELECT
        COUNT(*)                            AS n_total,
        SUM(is_fraud)                       AS n_fraud,
        SUM(is_fraud * amount)              AS amt_fraud
    FROM scored
),

-- Every achievable operating point is "review the top k by score". One
-- ordered pass with cumulative windows produces the entire sweep, at finer
-- resolution than any fixed threshold grid.
ranked AS (
    SELECT
        score,
        is_fraud,
        amount,
        ROW_NUMBER() OVER (ORDER BY score DESC, amount)              AS n_alerts,
        SUM(is_fraud) OVER (ORDER BY score DESC, amount
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                                     AS cum_tp,
        SUM(is_fraud * amount) OVER (ORDER BY score DESC, amount
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                                     AS cum_tp_amt,
        -- A threshold admits every row sharing that score, so only the last
        -- row of a tied block is a genuine cut point.
        LEAD(score) OVER (ORDER BY score DESC, amount)               AS next_score
    FROM scored
),

sweep AS (
    SELECT
        r.score                                          AS threshold,
        r.n_alerts,
        r.cum_tp                                         AS tp,
        r.n_alerts - r.cum_tp                            AS fp,
        t.n_fraud - r.cum_tp                             AS fn,
        t.n_total - r.n_alerts - (t.n_fraud - r.cum_tp)  AS tn,
        CAST(r.cum_tp AS DOUBLE) / r.n_alerts            AS precision,
        CAST(r.cum_tp AS DOUBLE) / t.n_fraud             AS recall,
        1.0 - CAST(r.cum_tp AS DOUBLE) / r.n_alerts      AS false_alarm_rate,
        CAST(r.n_alerts AS DOUBLE) / t.n_total           AS alert_rate,
        r.n_alerts * ({minutes_review} / 60.0)           AS review_hours,
        r.cum_tp_amt                                     AS fraud_caught_dollars,
        t.amt_fraud - r.cum_tp_amt                       AS fraud_missed_dollars,
        r.cum_tp_amt / t.amt_fraud                       AS dollar_recall,
        (r.n_alerts - r.cum_tp) * {cost_fp}              AS fp_cost,
        t.amt_fraud - r.cum_tp_amt                       AS fn_cost,
        (t.amt_fraud - r.cum_tp_amt)
            + (r.n_alerts - r.cum_tp) * {cost_fp}        AS total_cost,
        CASE WHEN r.next_score IS NULL OR r.next_score <> r.score
             THEN 1 ELSE 0 END                           AS valid_cut
    FROM ranked r
    CROSS JOIN totals t
)

SELECT
    threshold,
    n_alerts,
    tp, fp, fn, tn,
    precision,
    recall,
    false_alarm_rate,
    alert_rate,
    review_hours,
    fraud_caught_dollars,
    fraud_missed_dollars,
    dollar_recall,
    fp_cost,
    fn_cost,
    total_cost
FROM sweep
WHERE valid_cut = 1
ORDER BY n_alerts;
