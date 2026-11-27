"""
Automated Enterprise Excel Report Exporter (`excel_exporter.py`).

Generates a presentation-ready, multi-sheet `.xlsx` workbook via `openpyxl`:
1. `Executive_Summary`     — Metadata, KPI summary table, and native `BarChart`
                             comparing Before vs. After detections per class.
2. `Per_Class_Comparison`  — Per-class counts, deltas, %, and confidence shifts.
3. `Frame_Detail_Log`      — Granular per-image or per-sampled-frame log.
4. `Region_Counter_Log`    — Unique tracked objects entering the ROI polygon.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# Styling Palette (Carbon / Deep Navy Enterprise)
FILL_BANNER = PatternFill(start_color="090D16", end_color="090D16", fill_type="solid")
FILL_HEADER = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
FILL_SUBHEADER = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
FILL_ZEBRA = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
FILL_WHITE = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
FILL_TOTAL = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")

FONT_BANNER = Font(name="Calibri", size=14, bold=True, color="10B981")
FONT_SUBTITLE = Font(name="Calibri", size=10, italic=True, color="94A3B8")
FONT_HEADER = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
FONT_SECTION = Font(name="Calibri", size=11, bold=True, color="0F172A")
FONT_LABEL = Font(name="Calibri", size=10, bold=True, color="334155")
FONT_BODY = Font(name="Calibri", size=10, bold=False, color="0F172A")
FONT_BOLD = Font(name="Calibri", size=10, bold=True, color="0F172A")
FONT_POS = Font(name="Calibri", size=10, bold=True, color="059669")   # Emerald Green
FONT_NEG = Font(name="Calibri", size=10, bold=True, color="DC2626")   # Crimson Red
FONT_NEUTRAL = Font(name="Calibri", size=10, bold=False, color="64748B")

THIN_SIDE = Side(border_style="thin", color="CBD5E1")
MED_BOTTOM = Side(border_style="medium", color="0F172A")
BORDER_CELL = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)
BORDER_HEADER = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=MED_BOTTOM)

ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")


def _delta_font(val: float) -> Font:
    if val > 1e-6:
        return FONT_POS
    if val < -1e-6:
        return FONT_NEG
    return FONT_NEUTRAL


def _auto_fit_columns(ws: Any, min_width: int = 12, max_width: int = 44) -> None:
    """Auto-adjust worksheet column widths safely across merged & regular cells."""
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row_idx in range(1, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            # Skip merged banner rows on row 1-2 so Column A doesn't stretch to 90 chars
            if row_idx <= 2 and ws.title == "Executive_Summary":
                continue
            val_str = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(
            min_width, min(max_width, max_len + 4)
        )


def generate_enterprise_excel(
    report_data: Dict[str, Any],
    outputs_dir: Path,
) -> Path:
    """
    Build and save a styled 4-sheet `.xlsx` report in `outputs_dir`.
    Returns the absolute `Path` to the generated workbook.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    ts_slug = now.strftime("%Y%m%d_%H%M%S")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S")

    src_type = str(report_data.get("source_type", "session")).lower()
    filename = f"PasarVision_Analytics_{src_type}_{ts_slug}.xlsx"
    out_path = outputs_dir / filename

    summary = report_data.get("summary", {})
    per_class: List[Dict[str, Any]] = report_data.get("per_class_table", [])
    frame_log: List[Dict[str, Any]] = report_data.get("frame_detail_log", [])
    roi_log: List[Dict[str, Any]] = report_data.get("region_counter_log", [])

    wb = Workbook()

    # =========================================================================
    # SHEET 1: Executive_Summary
    # =========================================================================
    ws_exec = wb.active
    ws_exec.title = "Executive_Summary"
    ws_exec.views.sheetView[0].showGridLines = True

    # Title Banner
    ws_exec.merge_cells("A1:H1")
    ws_exec["A1"] = "PASARVISION AI — TRADITIONAL MARKET SOCIO-ECONOMIC ANALYTICS REPORT"
    ws_exec["A1"].font = FONT_BANNER
    ws_exec["A1"].fill = FILL_BANNER
    ws_exec["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_exec.row_dimensions[1].height = 30

    ws_exec.merge_cells("A2:H2")
    ws_exec["A2"] = (
        f"Generated: {ts_human}  |  Engine: YOLOv11 + ByteTrack + OpenCV Enhancement  |  "
        f"Mode: {report_data.get('mode', 'Market Analytics')}"
    )
    ws_exec["A2"].font = FONT_SUBTITLE
    ws_exec["A2"].fill = FILL_BANNER
    ws_exec["A2"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_exec.row_dimensions[2].height = 20

    # Metadata Block (Columns A-C, Rows 4-12)
    ws_exec.merge_cells("A4:C4")
    ws_exec["A4"] = "1. SESSION & CONFIGURATION METADATA"
    ws_exec["A4"].font = FONT_HEADER
    ws_exec["A4"].fill = FILL_HEADER
    ws_exec["A4"].alignment = ALIGN_LEFT
    ws_exec.row_dimensions[4].height = 24

    roi_pts = summary.get("roi_polygon", [])
    roi_str = ", ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in roi_pts) if roi_pts else "Default Aisle"

    metadata_rows = [
        ("Report Timestamp", ts_human),
        ("Analysis Mode", str(report_data.get("mode", "Live Detection"))),
        ("Input Media Source", str(report_data.get("source_name", "N/A"))),
        ("Active YOLOv11 Model", str(summary.get("model_badge", summary.get("model_name", "yolo11n.pt")))),
        ("Inference Hardware", str(summary.get("device_label", "CPU"))),
        ("Image Enhancement", str(summary.get("enhancement_description", "Original (None)"))),
        (
            "Inference Thresholds",
            f"Conf = {float(summary.get('conf_threshold', 0.35)):.2f}  |  "
            f"IoU NMS = {float(summary.get('iou_threshold', 0.45)):.2f}",
        ),
        ("Evaluated Frames / Files", int(summary.get("sampled_frames", max(1, len(frame_log))))),
        ("ROI Region Vertices", roi_str),
    ]

    for idx, (meta_key, meta_val) in enumerate(metadata_rows, start=5):
        fill = FILL_ZEBRA if idx % 2 == 1 else FILL_WHITE
        ws_exec.cell(row=idx, column=1, value=meta_key).font = FONT_LABEL
        ws_exec.cell(row=idx, column=1).fill = fill
        ws_exec.cell(row=idx, column=1).border = BORDER_CELL
        ws_exec.cell(row=idx, column=1).alignment = ALIGN_LEFT

        ws_exec.merge_cells(start_row=idx, start_column=2, end_row=idx, end_column=3)
        val_cell = ws_exec.cell(row=idx, column=2, value=meta_val)
        val_cell.font = FONT_BODY
        val_cell.fill = fill
        val_cell.alignment = ALIGN_LEFT
        for c in (2, 3):
            ws_exec.cell(row=idx, column=c).border = BORDER_CELL
            ws_exec.cell(row=idx, column=c).fill = fill
        ws_exec.row_dimensions[idx].height = 20

    # KPI Summary Table (Rows 15-24, Columns A-C)
    kpi_start = 15
    ws_exec.merge_cells(f"A{kpi_start}:C{kpi_start}")
    ws_exec[f"A{kpi_start}"] = "2. KEY PERFORMANCE INDICATORS (KPI SUMMARY)"
    ws_exec[f"A{kpi_start}"].font = FONT_HEADER
    ws_exec[f"A{kpi_start}"].fill = FILL_HEADER
    ws_exec.row_dimensions[kpi_start].height = 24

    hdr_row = kpi_start + 1
    for col_i, h_text in enumerate(["Metric Indicator", "Value", "Notes / Impact"], start=1):
        cell = ws_exec.cell(row=hdr_row, column=col_i, value=h_text)
        cell.font = FONT_HEADER
        cell.fill = FILL_SUBHEADER
        cell.border = BORDER_HEADER
        cell.alignment = ALIGN_CENTER if col_i == 2 else ALIGN_LEFT
    ws_exec.row_dimensions[hdr_row].height = 22

    b_tot = int(summary.get("before_total", 0))
    a_tot = int(summary.get("after_total", 0))
    d_cnt = int(summary.get("delta_count", a_tot - b_tot))
    d_pct_ratio = float(summary.get("delta_pct", 0.0)) / 100.0
    b_conf = float(summary.get("before_avg_conf", 0.0))
    a_conf = float(summary.get("after_avg_conf", 0.0))
    d_conf = float(summary.get("delta_conf", a_conf - b_conf))
    uniq_ids = int(summary.get("unique_tracked_total", a_tot))
    roi_entered = int(summary.get("region_unique_entered", summary.get("after_inside_roi", 0)))
    roi_peak = int(summary.get("after_inside_roi", 0))

    kpi_items = [
        ("Total Detections (Before / Raw)", b_tot, "#,##0", FONT_BODY, "Baseline unenhanced frames"),
        ("Total Detections (After / Active)", a_tot, "#,##0", FONT_BOLD, "Post-enhancement detections"),
        ("Detection Count Delta (Δ)", d_cnt, "+#,##0;-#,##0;0", _delta_font(d_cnt), "Net objects gained/lost"),
        ("Detection Relative Gain (Δ %)", d_pct_ratio, "+0.00%;-0.00%;0.00%", _delta_font(d_pct_ratio), "Relative recall change"),
        ("Avg Confidence (Before)", b_conf, "0.000", FONT_BODY, "Mean YOLOv11 box score (raw)"),
        ("Avg Confidence (After)", a_conf, "0.000", FONT_BOLD, "Mean YOLOv11 box score (active)"),
        ("Confidence Score Shift (Δ Conf)", d_conf, "+0.000;-0.000;0.000", _delta_font(d_conf), "Model certainty improvement"),
        ("Unique Tracked Objects (ByteTrack)", uniq_ids, "#,##0", FONT_BOLD, "Cumulative de-duplicated IDs"),
        ("ROI Region Counter (Unique Entered)", roi_entered, "#,##0", FONT_POS if roi_entered > 0 else FONT_BODY, f"Peak Simultaneous Inside ROI: {roi_peak}"),
    ]

    for offset, (m_name, m_val, num_fmt, val_font, note_txt) in enumerate(kpi_items, start=1):
        r = hdr_row + offset
        fill = FILL_ZEBRA if offset % 2 == 1 else FILL_WHITE
        c1 = ws_exec.cell(row=r, column=1, value=m_name)
        c2 = ws_exec.cell(row=r, column=2, value=m_val)
        c3 = ws_exec.cell(row=r, column=3, value=note_txt)

        c1.font, c1.fill, c1.border, c1.alignment = FONT_LABEL, fill, BORDER_CELL, ALIGN_LEFT
        c2.font, c2.fill, c2.border, c2.alignment = val_font, fill, BORDER_CELL, ALIGN_RIGHT
        c2.number_format = num_fmt
        c3.font, c3.fill, c3.border, c3.alignment = FONT_NEUTRAL, fill, BORDER_CELL, ALIGN_LEFT
        ws_exec.row_dimensions[r].height = 20

    # Chart Reference Table on Executive_Summary (Row 28+) + Embedded Native BarChart at E4
    chart_tbl_start = hdr_row + len(kpi_items) + 3
    ws_exec.merge_cells(f"A{chart_tbl_start}:C{chart_tbl_start}")
    ws_exec[f"A{chart_tbl_start}"] = "3. CLASS DETECTION SUMMARY (CHART DATA SOURCE)"
    ws_exec[f"A{chart_tbl_start}"].font = FONT_HEADER
    ws_exec[f"A{chart_tbl_start}"].fill = FILL_HEADER
    ws_exec.row_dimensions[chart_tbl_start].height = 22

    chart_hdr = chart_tbl_start + 1
    for ci, ch_name in enumerate(["Class", "Before Count", "After Count"], start=1):
        hc = ws_exec.cell(row=chart_hdr, column=ci, value=ch_name)
        hc.font = FONT_HEADER
        hc.fill = FILL_SUBHEADER
        hc.border = BORDER_HEADER
        hc.alignment = ALIGN_LEFT if ci == 1 else ALIGN_RIGHT

    chart_rows = per_class if per_class else [
        {"class_name": "No Detections", "before_count": 0, "after_count": 0}
    ]
    for idx_c, crow in enumerate(chart_rows, start=1):
        rr = chart_hdr + idx_c
        fill = FILL_ZEBRA if idx_c % 2 == 1 else FILL_WHITE
        ca = ws_exec.cell(row=rr, column=1, value=crow["class_name"])
        cb = ws_exec.cell(row=rr, column=2, value=int(crow["before_count"]))
        cc = ws_exec.cell(row=rr, column=3, value=int(crow["after_count"]))
        ca.font, ca.fill, ca.border, ca.alignment = FONT_LABEL, fill, BORDER_CELL, ALIGN_LEFT
        cb.font, cb.fill, cb.border, cb.alignment = FONT_BODY, fill, BORDER_CELL, ALIGN_RIGHT
        cc.font, cc.fill, cc.border, cc.alignment = FONT_BOLD, fill, BORDER_CELL, ALIGN_RIGHT
        cb.number_format = "#,##0"
        cc.number_format = "#,##0"

    chart_end_row = chart_hdr + len(chart_rows)

    # Native OpenPyXL Grouped BarChart
    bar_chart = BarChart()
    bar_chart.type = "col"
    bar_chart.style = 10
    bar_chart.grouping = "clustered"
    bar_chart.overlap = -15
    bar_chart.title = "Before vs. After Enhancement Detections by Class"
    bar_chart.y_axis.title = "Detected Objects"
    bar_chart.x_axis.title = "Market Socio-Economic Class"
    bar_chart.width = 18
    bar_chart.height = 12

    data_ref = Reference(
        ws_exec,
        min_col=2,
        max_col=3,
        min_row=chart_hdr,
        max_row=chart_end_row,
    )
    cats_ref = Reference(
        ws_exec,
        min_col=1,
        min_row=chart_hdr + 1,
        max_row=chart_end_row,
    )
    bar_chart.add_data(data_ref, titles_from_data=True)
    bar_chart.set_categories(cats_ref)

    # Style series colors: Slate for Before, Emerald for After
    if len(bar_chart.series) >= 2:
        bar_chart.series[0].graphicalProperties.solidFill = "64748B"
        bar_chart.series[1].graphicalProperties.solidFill = "10B981"

    ws_exec.add_chart(bar_chart, "E4")
    ws_exec.freeze_panes = "A4"
    _auto_fit_columns(ws_exec, min_width=15, max_width=40)

    # =========================================================================
    # SHEET 2: Per_Class_Comparison
    # =========================================================================
    ws_cls = wb.create_sheet(title="Per_Class_Comparison")
    cls_headers = [
        "Class",
        "Before Count",
        "After Count",
        "Delta Count",
        "Delta %",
        "Before Avg Conf",
        "After Avg Conf",
        "Conf Delta",
    ]
    for col_i, h_title in enumerate(cls_headers, start=1):
        cell = ws_cls.cell(row=1, column=col_i, value=h_title)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.border = BORDER_HEADER
        cell.alignment = ALIGN_LEFT if col_i == 1 else ALIGN_RIGHT
    ws_cls.row_dimensions[1].height = 25

    if per_class:
        for r_idx, item in enumerate(per_class, start=2):
            fill = FILL_ZEBRA if r_idx % 2 == 0 else FILL_WHITE
            dc = int(item["delta_count"])
            dp_ratio = float(item["delta_pct"]) / 100.0
            cd = float(item["conf_delta"])

            vals = [
                (item["class_name"], None, FONT_LABEL, ALIGN_LEFT),
                (int(item["before_count"]), "#,##0", FONT_BODY, ALIGN_RIGHT),
                (int(item["after_count"]), "#,##0", FONT_BOLD, ALIGN_RIGHT),
                (dc, "+#,##0;-#,##0;0", _delta_font(dc), ALIGN_RIGHT),
                (dp_ratio, "+0.00%;-0.00%;0.00%", _delta_font(dp_ratio), ALIGN_RIGHT),
                (float(item["before_avg_conf"]), "0.000", FONT_BODY, ALIGN_RIGHT),
                (float(item["after_avg_conf"]), "0.000", FONT_BOLD, ALIGN_RIGHT),
                (cd, "+0.000;-0.000;0.000", _delta_font(cd), ALIGN_RIGHT),
            ]
            for c_i, (v, nfmt, fnt, algn) in enumerate(vals, start=1):
                c = ws_cls.cell(row=r_idx, column=c_i, value=v)
                c.font, c.fill, c.border, c.alignment = fnt, fill, BORDER_CELL, algn
                if nfmt:
                    c.number_format = nfmt
            ws_cls.row_dimensions[r_idx].height = 20

        # Total / Average Summary Row
        tot_r = len(per_class) + 2
        tot_vals = [
            ("TOTAL / MEAN", None, FONT_BOLD, ALIGN_LEFT),
            (b_tot, "#,##0", FONT_BOLD, ALIGN_RIGHT),
            (a_tot, "#,##0", FONT_BOLD, ALIGN_RIGHT),
            (d_cnt, "+#,##0;-#,##0;0", _delta_font(d_cnt), ALIGN_RIGHT),
            (d_pct_ratio, "+0.00%;-0.00%;0.00%", _delta_font(d_pct_ratio), ALIGN_RIGHT),
            (b_conf, "0.000", FONT_BOLD, ALIGN_RIGHT),
            (a_conf, "0.000", FONT_BOLD, ALIGN_RIGHT),
            (d_conf, "+0.000;-0.000;0.000", _delta_font(d_conf), ALIGN_RIGHT),
        ]
        for c_i, (v, nfmt, fnt, algn) in enumerate(tot_vals, start=1):
            c = ws_cls.cell(row=tot_r, column=c_i, value=v)
            c.font, c.fill, c.border, c.alignment = fnt, FILL_TOTAL, BORDER_HEADER, algn
            if nfmt:
                c.number_format = nfmt
        ws_cls.row_dimensions[tot_r].height = 22
    else:
        c = ws_cls.cell(row=2, column=1, value="No detections above active confidence threshold")
        c.font = FONT_NEUTRAL

    ws_cls.freeze_panes = "A2"
    _auto_fit_columns(ws_cls)

    # =========================================================================
    # SHEET 3: Frame_Detail_Log
    # =========================================================================
    ws_frm = wb.create_sheet(title="Frame_Detail_Log")
    frm_headers = [
        "Frame/File",
        "Timestamp (s)",
        "Before Detections",
        "After Detections",
        "Before Avg Conf",
        "After Avg Conf",
        "Inside ROI Count",
    ]
    for col_i, h_title in enumerate(frm_headers, start=1):
        cell = ws_frm.cell(row=1, column=col_i, value=h_title)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.border = BORDER_HEADER
        cell.alignment = ALIGN_LEFT if col_i == 1 else ALIGN_RIGHT
    ws_frm.row_dimensions[1].height = 25

    if frame_log:
        for r_idx, f_item in enumerate(frame_log, start=2):
            fill = FILL_ZEBRA if r_idx % 2 == 0 else FILL_WHITE
            f_vals = [
                (str(f_item.get("frame_or_file", f"Frame #{r_idx - 1}")), None, FONT_LABEL, ALIGN_LEFT),
                (float(f_item.get("timestamp_sec", 0.0)), "0.00", FONT_BODY, ALIGN_RIGHT),
                (int(f_item.get("before_detections", 0)), "#,##0", FONT_BODY, ALIGN_RIGHT),
                (int(f_item.get("after_detections", 0)), "#,##0", FONT_BOLD, ALIGN_RIGHT),
                (float(f_item.get("before_avg_conf", 0.0)), "0.000", FONT_BODY, ALIGN_RIGHT),
                (float(f_item.get("after_avg_conf", 0.0)), "0.000", FONT_BOLD, ALIGN_RIGHT),
                (int(f_item.get("inside_roi_count", 0)), "#,##0", FONT_POS if int(f_item.get("inside_roi_count", 0)) > 0 else FONT_BODY, ALIGN_RIGHT),
            ]
            for c_i, (v, nfmt, fnt, algn) in enumerate(f_vals, start=1):
                c = ws_frm.cell(row=r_idx, column=c_i, value=v)
                c.font, c.fill, c.border, c.alignment = fnt, fill, BORDER_CELL, algn
                if nfmt:
                    c.number_format = nfmt
            ws_frm.row_dimensions[r_idx].height = 19
    else:
        ws_frm.cell(row=2, column=1, value="No frame log entries recorded").font = FONT_NEUTRAL

    ws_frm.freeze_panes = "A2"
    _auto_fit_columns(ws_frm)

    # =========================================================================
    # SHEET 4: Region_Counter_Log
    # =========================================================================
    ws_roi = wb.create_sheet(title="Region_Counter_Log")
    roi_headers = [
        "Track ID",
        "Class",
        "First Entered (s)",
        "Confidence",
    ]
    for col_i, h_title in enumerate(roi_headers, start=1):
        cell = ws_roi.cell(row=1, column=col_i, value=h_title)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.border = BORDER_HEADER
        cell.alignment = ALIGN_CENTER if col_i == 1 else (ALIGN_LEFT if col_i == 2 else ALIGN_RIGHT)
    ws_roi.row_dimensions[1].height = 25

    if roi_log:
        for r_idx, r_item in enumerate(roi_log, start=2):
            fill = FILL_ZEBRA if r_idx % 2 == 0 else FILL_WHITE
            r_vals = [
                (str(r_item.get("track_id", f"#{r_idx - 1}")), None, FONT_BOLD, ALIGN_CENTER),
                (str(r_item.get("class_name", "object")), None, FONT_LABEL, ALIGN_LEFT),
                (float(r_item.get("first_entered_sec", 0.0)), "0.00", FONT_BODY, ALIGN_RIGHT),
                (float(r_item.get("confidence", 0.0)), "0.000", FONT_POS, ALIGN_RIGHT),
            ]
            for c_i, (v, nfmt, fnt, algn) in enumerate(r_vals, start=1):
                c = ws_roi.cell(row=r_idx, column=c_i, value=v)
                c.font, c.fill, c.border, c.alignment = fnt, fill, BORDER_CELL, algn
                if nfmt:
                    c.number_format = nfmt
            ws_roi.row_dimensions[r_idx].height = 19
    else:
        msg_cell = ws_roi.cell(
            row=2,
            column=1,
            value="No tracked objects entered the ROI polygon during this session",
        )
        msg_cell.font = FONT_NEUTRAL

    ws_roi.freeze_panes = "A2"
    _auto_fit_columns(ws_roi, min_width=16)

    wb.save(str(out_path))
    return out_path
