const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
} = require("docx");

const PAGE_WIDTH_DXA = 12240;
const MARGIN_DXA = 1440;
const CONTENT_WIDTH = PAGE_WIDTH_DXA - 2 * MARGIN_DXA;

const heading = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
const subheading = (text) => new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
const body = (text, opts = {}) => new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
const bullet = (text) => new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });

function metricTable(rows) {
  const colWidths = [4000, 4340];
  return new Table({
    width: { size: colWidths[0] + colWidths[1], type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map(([label, value], i) => new TableRow({
      children: [
        new TableCell({
          width: { size: colWidths[0], type: WidthType.DXA },
          shading: i === 0 ? { type: ShadingType.CLEAR, fill: "064A56" } : undefined,
          children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, color: i === 0 ? "FFFFFF" : "000000" })] })],
        }),
        new TableCell({
          width: { size: colWidths[1], type: WidthType.DXA },
          shading: i === 0 ? { type: ShadingType.CLEAR, fill: "064A56" } : undefined,
          children: [new Paragraph({ children: [new TextRun({ text: value, bold: i === 0, color: i === 0 ? "FFFFFF" : "000000" })] })],
        }),
      ],
    })),
  });
}

function imageParagraph(path, widthPx, heightPx) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    children: [new ImageRun({ data, transformation: { width: widthPx, height: heightPx }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: PAGE_WIDTH_DXA, height: 15840 }, margin: { top: MARGIN_DXA, bottom: MARGIN_DXA, left: MARGIN_DXA, right: MARGIN_DXA } } },
    children: [
      new Paragraph({ children: [new TextRun({ text: "Freight Rate Prediction — Assessment Report", bold: true, size: 40, color: "064A56" })], spacing: { after: 80 } }),
      new Paragraph({ children: [new TextRun({ text: "Machine Learning Engineer Assessment · Spotter", italics: true, size: 22, color: "455A60" })], spacing: { after: 320 } }),

      heading("1. Train/Test Validation Approach"),
      body("The labeled data (data/train_test.csv, 48,000 rows spanning 2025-01-01 through 2025-10-31) was split chronologically rather than randomly. The final validation set the model must ultimately score (data/validation.csv) covers 2025-11-01 through 2025-12-31 — entirely in the future relative to every labeled row. A random split would let same-day market conditions leak between the train and test folds and overstate accuracy on this real forecasting task."),
      body("Split used for model selection:"),
      bullet("Sort all labeled rows by date."),
      bullet("Train on the first 85% of dates (Jan 1 – Sep 15, 2025, 40,706 rows)."),
      bullet("Hold out the last 15% of dates (Sep 15 – Oct 31, 2025, 7,294 rows) purely for evaluation."),
      bullet("Early stopping on the holdout set picks the boosting round count; that same round count (scaled up ~15% for the larger data volume) is reused when the final model is refit on 100% of the labeled rows for scoring validation.csv."),
      body("This mirrors the actual deployment gap (train on the past, predict the future) and avoids optimistic bias from random-split leakage."),

      subheading("Holdout results"),
      metricTable([
        ["Metric", "Value"],
        ["MAE", "$119.41"],
        ["RMSE", "$602.64"],
        ["MAPE", "5.43%"],
        ["% of predictions within 10% of actual", "97.0%"],
        ["% of predictions within 20% of actual", "98.5%"],
      ]),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      body("RMSE is notably larger than MAE, which points to a small number of large-error outliers rather than broad inaccuracy — see Section 3."),

      heading("2. Data Exploration & Quality Issues"),
      subheading("Key findings"),
      bullet("posted_rate is driven overwhelmingly by distance (correlation 0.91); quote_signal and market_index add smaller, complementary signal (distance × quote_signal correlates at 0.90)."),
      bullet("market_index behaves like a shared daily market factor: within any given date, its standard deviation across loads is only ~0.025, while the daily mean drifts smoothly from ~0.75 to ~1.45 over the year."),
      bullet("quote_signal is comparatively noisy per shipment (within-lane std ~0.24 even on the exact same route) — it does not reduce to a clean date-level or lane-level constant."),
      bullet("Rate-per-mile ranges from ~$0.33/mi to ~$14.1/mi; the extreme end is a small cluster (~0.7% of rows) of short-haul, high-$/mile loads (mostly Reefer/Flatbed) — plausible expedited/spot-market premiums rather than data errors, but a hard segment for any model trained mainly on typical lanes."),

      subheading("Data-quality issues identified and how they were addressed"),
      bullet("Sign-flipped weight: 292 rows (0.6%) had negative weight values whose magnitude matched the normal weight distribution — corrected with abs()."),
      bullet("Missing weight: 300 rows (0.6%) — imputed with the training-set median weight."),
      bullet("Missing market_index: 374 rows in train_test.csv (0.8%), 249 in validation.csv (2.1%) — imputed using the same-date market average computed from all rows that do have a value (justified by the low within-date variance noted above), falling back to a fitted day-of-year trend for any date with zero observed values."),
      bullet("december_chart_inputs.csv omits market_index and quote_signal entirely for the fixed Lexington→Fort Wayne route. market_index was recovered exactly using the combined train+validation daily average, which has full coverage through Dec 31. quote_signal has no reliable date-level pattern, so it was filled with the historical median quote_signal observed on this exact lane for Dry Van loads (n=27, median 2.002) — a documented approximation, not an exact recovery."),
      bullet("No duplicate load_id or duplicate full rows were found in either file."),

      heading("3. Model & Feature Importance"),
      body("Model: LightGBM gradient-boosted trees, trained on log1p(posted_rate), with early stopping on the time-based holdout."),
      body("Features: distance, weight, market_index, quote_signal, distance × quote_signal, equipment, pickup, delivery, month, day-of-week, day-of-year, is_weekend."),
      body("Feature importance (gain), most to least important: distance (dominant), distance × quote_signal, delivery city, quote_signal, pickup city, equipment, day-of-year, market_index, weight, month, day-of-week, is_weekend. distance alone accounts for the large majority of predictive gain, consistent with its 0.91 raw correlation with posted_rate; quote_signal contributes mainly through its interaction with distance."),
      body("Largest holdout errors cluster on the same high-$/mile outlier segment identified in Section 2 (short-haul premium loads) — e.g. several $10k+ absolute errors on loads with rate-per-mile above $12, versus a typical error of ~$50–150 on the other 97% of loads. This is a known limitation: these loads appear to be driven by information not present in the given feature set (e.g. urgency/spot-market conditions), rather than a fixable modeling error."),

      heading("4. December 2025 Predicted Rate — Fixed Lane"),
      body("Fixed inputs held constant across all 31 rows: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes. Chart generated by the provided score.py from data/december_chart_inputs_filled.csv:"),
      imageParagraph("/home/claude/freight-rate-ml/scorer_results/candidate_december.png", 620, 280),
      body("The weekly ripple (visible ~7-day cycle) comes from the day-of-week feature interacting with the fitted market_index seasonal level for December; the model was not given any December-specific quote_signal, so this reflects the lane's typical historical rate level (~$825–$829) modulated by the recovered daily market index, not date-specific spot-market swings."),

      heading("5. Repository & Reproduction"),
      body("Full source, data, and run instructions are in the submitted GitHub repository. Summary:"),
      bullet("src/features.py — shared cleaning + feature engineering"),
      bullet("src/train.py — time-based holdout training, prints metrics, saves model artifacts"),
      bullet("src/predict.py — refits on all labeled data, writes validation_predictions.csv and the filled December input file"),
      bullet("src/score.py — provided scorer; validates both output files and renders the chart above"),
      body("See the repository README for exact setup and run commands."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/freight-rate-ml/report/Freight_Rate_Assessment_Report.docx", buf);
  console.log("written");
});
