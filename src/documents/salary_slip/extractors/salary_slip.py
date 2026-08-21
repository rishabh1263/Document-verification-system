"""
Generic salary-slip extractor.

Goals:
- No company/template-specific conditions.
- Works with native PDF words or OCR words with bounding boxes.
- Uses aliases + spatial matching + validation + text fallback.
- Keeps the output schema stable for downstream consistency checks.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .base_extractor import BaseExtractor

DEBUG_BUILD_MARKER = "SALARY_SLIP_MERGED_BASIC_V11_2026_08_04"


class SalarySlipExtractor(BaseExtractor):
    doc_type = "salary_slip"

    FIELD_ALIASES = {
        "employee_name": [
            "employee name", "emp name", "name of employee", "staff name",
            "associate name", "worker name", "employee full name",
        ],
        "employee_id": [
            "employee id", "employee no", "employee number", "employee code",
            "emp id", "emp no", "emp number", "emp code", "staff id",
            "staff code", "associate id", "personnel no", "personnel number",
            "worker id",
        ],
        "designation": [
            "designation", "job title", "position", "role", "grade designation",
        ],
        "pay_period": [
            "pay period", "pay month", "salary month", "salary period",
            "month of", "salary for", "for the month of",
        ],
        "pan": ["pan", "pan no", "pan number"],
        "bank_account": [
            "bank account", "bank account no", "bank account number",
            "account no", "account number", "bank a/c", "bank a/c no",
            "bank ac no",
        ],
        "basic_pay": ["basic pay", "basic salary", "basic"],
        "gross_pay": [
            "gross pay", "gross salary", "gross earnings", "total earnings",
            "total earning", "gross amount", "total emoluments",
        ],
        "total_deductions": [
            "total deductions", "total deduction", "deductions total",
            "total contribution and deductions", "total contribution & deductions",
        ],
        "net_pay": [
            "net pay", "net salary", "net payable", "net salary payable",
            "take home", "take home salary", "take home pay", "amount payable",
            "net amount", "net payable amount",
        ],
    }

    FIELD_TYPES = {
        "employee_name": "name",
        "employee_id": "employee_id",
        "designation": "designation",
        "pay_period": "period",
        "pan": "pan",
        "bank_account": "account",
        "basic_pay": "money",
        "gross_pay": "money",
        "total_deductions": "money",
        "net_pay": "money",
    }

    MONEY_FIELDS = {"basic_pay", "gross_pay", "total_deductions", "net_pay"}

    BLOCKED_TEXT = {
        "employee", "employee name", "employee id", "employee number",
        "employee code", "designation", "department", "location", "date",
        "date joined", "date of joining", "payment", "payment mode", "uan",
        "pf", "pf number", "pan", "pan number", "bank", "bank account",
        "earnings", "deductions", "salary", "salary details", "payslip",
        "pay slip", "gross pay", "gross salary", "net pay", "net salary",
        "total earnings", "total deductions", "working days", "lop days",
    }

    MONTH_RE = re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[\s\-/,]*(?:19|20)\d{2}\b",
        re.IGNORECASE,
    )

    PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)

    MONEY_RE = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:â‚¹|Rs\.?|INR)?\s*"
        r"("
        r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?"
        r"|"
        r"\d{4,}(?:\.\d{1,2})?"
        r")"
        r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> Dict[str, Any]:
        """Text-only fallback for PDFs/OCR where layout boxes are unavailable."""
        text = self._normalize_text(text)

        fields = self._empty_fields()

        fields["employee_name"] = self._text_field(text, "employee_name")
        fields["employee_id"] = self._text_field(text, "employee_id")
        fields["designation"] = self._text_field(text, "designation")
        fields["pay_period"] = self._extract_period(text)
        fields["pan"] = self._extract_pan(text)
        fields["bank_account"] = self._text_field(text, "bank_account")

        for field in self.MONEY_FIELDS:
            fields[field] = self._text_money_field(text, field)

        self._infer_money_fields(fields)
        self._sanitize_fields(fields)
        fields["_numeric"] = self._to_numeric_summary(fields)
        return fields

    def extract_with_layout(self, text: str, words) -> Dict[str, Any]:
        """
        Generic layout extraction.

        It does not assume fixed X/Y coordinates. It builds visual rows,
        finds label aliases, then scores candidate values on the same row
        and nearby rows/columns.
        """
        fields = self.extract(text)

        if not words:
            return fields

        rows = self._group_words_into_rows(words)

        layout_id, layout_name = self._extract_combined_id_name_from_rows(rows)
        if layout_id:
            fields["employee_id"] = layout_id
        if layout_name:
            fields["employee_name"] = layout_name

        for field, aliases in self.FIELD_ALIASES.items():
            # Do not overwrite a strong combined ID/Name extraction with a
            # weaker generic proximity candidate.
            if field == "employee_id" and layout_id:
                continue
            if field == "employee_name" and layout_name:
                continue
            candidate = self._best_layout_candidate(
                rows=rows,
                field=field,
                aliases=aliases,
            )

            if candidate is not None:
                fields[field] = candidate

        # V4: reconstruct monetary fields from page geometry. Strong spatial
        # results override weak nearest-label candidates.
        geometry_money = self._extract_money_geometry(rows)
        for money_field, money_value in geometry_money.items():
            if money_field == "basic_pay":
                # V11: Basic geometry is authoritative. None is meaningful:
                # it prevents a weak regex/proximity fallback from injecting
                # a rate/day value into the final response.
                fields[money_field] = money_value
            elif money_value is not None:
                fields[money_field] = money_value

        # Global fallbacks that do not depend on labels.
        if not self._valid_value(fields.get("pan"), "pan"):
            fields["pan"] = self._extract_pan(text)

        if not self._valid_value(fields.get("pay_period"), "period"):
            fields["pay_period"] = self._extract_period(text)

        # Common combined identity form:
        # ID / Name : 000205 / Vaibhav Prakash Torase
        combined_id, combined_name = self._extract_combined_id_name(text)
        if not self._valid_value(fields.get("employee_id"), "employee_id") and combined_id:
            fields["employee_id"] = combined_id
        if not self._valid_value(fields.get("employee_name"), "name") and combined_name:
            fields["employee_name"] = combined_name

        self._infer_money_fields(fields)
        self._sanitize_fields(fields)
        fields["_numeric"] = self._to_numeric_summary(fields)
        return fields

    def _extract_money_geometry(self, rows: List[List[Any]]) -> Dict[str, Optional[str]]:
        """
        Reconstruct the salary table from geometry instead of fixed template
        coordinates.

        Strategy:
        - locate the table vertically from Basic / deduction / Net labels;
        - infer the deduction-description column from PF/ESIC/etc.;
        - cluster monetary X positions on the earnings side;
        - treat the right-most earnings cluster as Earned Amount;
        - infer the deduction-amount cluster from values to the right of
          deduction descriptions;
        - use arithmetic only after spatial extraction.

        Coordinates are inferred independently for every page.
        """
        result = {
            "basic_pay": None,
            "gross_pay": None,
            "total_deductions": None,
            "net_pay": None,
        }

        if not rows:
            return result

        all_words = [w for row in rows for w in row]
        if not all_words:
            return result

        page_width = max(float(w.bbox[2]) for w in all_words)
        heights = [
            max(1.0, float(w.bbox[3]) - float(w.bbox[1]))
            for w in all_words
        ]
        typical_h = sorted(heights)[len(heights) // 2]

        def cy(word):
            return (float(word.bbox[1]) + float(word.bbox[3])) / 2.0

        def cx(word):
            return (float(word.bbox[0]) + float(word.bbox[2])) / 2.0

        def token_money(word):
            raw = str(word.text).strip()
            raw = raw.replace("â‚¹", "").replace("Rs.", "").replace("Rs", "").strip()
            raw = raw.strip("[](){}|;:")
            if not re.fullmatch(r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{2})", raw):
                return None
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                return None
            if value < 1:
                return None
            return raw, value

        basic_words = [
            w for w in all_words
            if re.search(
                r"(?<![a-z])basic(?![a-z])",
                self._norm(str(w.text)).strip(),
                flags=re.IGNORECASE,
            )
        ]

        net_words = [
            w for w in all_words
            if self._norm(str(w.text)).strip() in {"net", "netpayable"}
            or "netpayable" in self._norm(str(w.text)).replace(" ", "")
        ]

        deduction_terms = {
            "pf", "esic", "esi", "pt", "tds", "lwf",
            "canteen", "deduction", "deductions",
        }
        deduction_words = [
            w for w in all_words
            if any(
                term == self._norm(str(w.text)).strip()
                or term in self._norm(str(w.text)).strip()
                for term in deduction_terms
            )
        ]

        # Vertical table range.
        y_candidates = [cy(w) for w in basic_words + deduction_words]
        if not y_candidates:
            return result

        table_top = min(y_candidates) - typical_h * 2.5
        table_bottom = (
            min(cy(w) for w in net_words) + typical_h
            if net_words
            else max(cy(w) for w in all_words)
        )

        table_words = [
            w for w in all_words
            if table_top <= cy(w) <= table_bottom
        ]

        # Deduction descriptions usually sit to the right of earnings.
        ded_desc_xs = [
            cx(w) for w in deduction_words
            if cx(w) > page_width * 0.50
        ]
        ded_desc_x = (
            sorted(ded_desc_xs)[len(ded_desc_xs) // 2]
            if ded_desc_xs
            else page_width * 0.70
        )

        # Collect monetary tokens in the table.
        money = []
        for w in table_words:
            parsed = token_money(w)
            if parsed is None:
                continue
            raw, value = parsed

            # Long identifiers are not money.
            digits = re.sub(r"\D", "", raw)
            if len(digits) >= 10 and "." not in raw and "," not in raw:
                continue

            money.append({
                "word": w,
                "raw": raw,
                "value": value,
                "x": cx(w),
                "y": cy(w),
            })

        if not money:
            return result

        def cluster_x(items, tolerance):
            clusters = []
            for item in sorted(items, key=lambda m: m["x"]):
                best = None
                best_dist = None
                for cluster in clusters:
                    dist = abs(item["x"] - cluster["center"])
                    if dist <= tolerance and (best_dist is None or dist < best_dist):
                        best = cluster
                        best_dist = dist
                if best is None:
                    clusters.append({"center": item["x"], "items": [item]})
                else:
                    best["items"].append(item)
                    best["center"] = sum(i["x"] for i in best["items"]) / len(best["items"])
            return clusters

        # Earnings side: ignore attendance/day/rate area on the left.
        earning_money = [
            m for m in money
            if page_width * 0.45 <= m["x"] < ded_desc_x - page_width * 0.015
        ]
        earning_clusters = cluster_x(
            earning_money,
            tolerance=max(18.0, page_width * 0.035),
        )
        earning_clusters = [
            c for c in earning_clusters
            if len(c["items"]) >= 1
        ]

        # The earned-amount column is the right-most stable monetary column
        # before deduction descriptions.
        earned_cluster = max(
            earning_clusters,
            key=lambda c: c["center"],
            default=None,
        )

        # Deduction amount values are to the right of deduction descriptions.
        deduction_money = [
            m for m in money
            if m["x"] > ded_desc_x + page_width * 0.05
        ]
        deduction_clusters = cluster_x(
            deduction_money,
            tolerance=max(18.0, page_width * 0.035),
        )
        deduction_cluster = max(
            deduction_clusters,
            key=lambda c: len(c["items"]),
            default=None,
        )

        # --------------------------------------------------------
        # BASIC PAY = Earned Amount spatially aligned with Basic.
        # --------------------------------------------------------
        if basic_words:
            basic_y = min(cy(w) for w in basic_words)

            # ----------------------------------------------------
            # V7 FULL-TABLE COLUMN INFERENCE FOR BASIC PAY
            # ----------------------------------------------------
            # The previous versions could confuse:
            #
            #     Rate       -> 1,458.00
            #     Earned     -> 18,xxx.xx
            #
            # because both values sit close to the Basic label.
            #
            # V7 infers monetary columns from ALL earnings rows on
            # the page. A rate column tends to contain smaller,
            # repeated values, while an earned-amount column tends
            # to contain larger payroll amounts. We therefore score
            # whole columns first, then use the winning column only
            # for the Basic row.
            # ----------------------------------------------------

            body_money = [
                m for m in earning_money
                if table_top <= m["y"] <= table_bottom
                and m["value"] >= 1
            ]

            column_clusters = cluster_x(
                body_money,
                tolerance=max(16.0, page_width * 0.028),
            )

            def median(values):
                if not values:
                    return 0.0
                values = sorted(values)
                n = len(values)
                mid = n // 2
                if n % 2:
                    return values[mid]
                return (values[mid - 1] + values[mid]) / 2.0

            scored_columns = []

            for cluster in column_clusters:
                items = cluster["items"]
                values = [m["value"] for m in items]

                if not values:
                    continue

                med = median(values)
                large_ratio = sum(v >= 5000 for v in values) / len(values)
                small_ratio = sum(v < 5000 for v in values) / len(values)

                score = 0.0

                # Earned salary columns normally contain several
                # substantial payroll amounts.
                score += large_ratio * 100.0

                # Median magnitude helps distinguish Earned Amount
                # from Rate/Days columns without fixed coordinates.
                if med >= 10000:
                    score += 60.0
                elif med >= 5000:
                    score += 35.0
                elif med >= 2500:
                    score += 10.0

                # Strongly penalize columns dominated by small values.
                score -= small_ratio * 45.0

                # Prefer stable/repeated table columns.
                score += min(len(items), 8) * 3.0

                # Earnings amount must remain left of deductions.
                if cluster["center"] >= ded_desc_x:
                    score -= 100.0

                scored_columns.append(
                    {
                        "center": cluster["center"],
                        "items": items,
                        "score": score,
                        "median": med,
                        "large_ratio": large_ratio,
                    }
                )

            earned_amount_column = None

            if scored_columns:
                earned_amount_column = max(
                    scored_columns,
                    key=lambda c: (
                        c["score"],
                        c["median"],
                        c["center"],
                    ),
                )

            if earned_amount_column is not None:
                # OCR may place the amount on the next text row, so
                # use a narrow vertical band around Basic.
                basic_candidates = [
                    m for m in body_money
                    if abs(m["y"] - basic_y) <= typical_h * 2.6
                    and abs(
                        m["x"] - earned_amount_column["center"]
                    ) <= max(30.0, page_width * 0.045)
                ]

                if basic_candidates:
                    chosen = min(
                        basic_candidates,
                        key=lambda m: (
                            # V9: visual-row alignment is more important
                            # than tiny X differences inside the same
                            # inferred monetary column.
                            abs(m["y"] - basic_y),
                            abs(m["x"] - earned_amount_column["center"]),
                        ),
                    )

                    # Store the spatial Basic candidate first.
                    # V11 validates Basic <= Gross only after Gross has
                    # actually been resolved later in this function.
                    result["basic_pay"] = chosen["raw"]

        # --------------------------------------------------------
        # NET PAY = explicit Net Payable value, regardless of table columns.
        # --------------------------------------------------------
        if net_words:
            for net_word in sorted(net_words, key=cy):
                same_band = [
                    m for m in money
                    if m["x"] > cx(net_word)
                    and abs(m["y"] - cy(net_word)) <= typical_h * 1.8
                ]
                if same_band:
                    chosen = max(same_band, key=lambda m: m["x"])
                    result["net_pay"] = chosen["raw"]
                    break

        # --------------------------------------------------------
        # TOTAL DEDUCTIONS
        # Prefer an explicit TOTAL row. Otherwise sum the inferred
        # deduction-amount column.
        # --------------------------------------------------------
        explicit_ded_total = None
        for row in rows:
            row_text = " ".join(str(w.text) for w in row)
            norm = self._norm(row_text)
            if "total deduction" not in norm and not (
                "total" in norm and "deduct" in norm
            ):
                continue
            row_money = []
            for w in row:
                parsed = token_money(w)
                if parsed:
                    row_money.append((cx(w), parsed[0], parsed[1]))
            right = [m for m in row_money if m[0] > ded_desc_x]
            if right:
                explicit_ded_total = max(right, key=lambda m: m[0])[1]
                break

        if explicit_ded_total:
            result["total_deductions"] = explicit_ded_total
        elif deduction_cluster:
            # Keep only values reasonably close to the inferred deduction
            # amount column and inside the salary table body.
            values = [
                m["value"] for m in deduction_cluster["items"]
                if table_top <= m["y"] <= table_bottom
            ]
            if values:
                result["total_deductions"] = self._format_money(sum(values))

        # --------------------------------------------------------
        # GROSS PAY
        # Prefer explicit Gross/Total Earnings. If absent, gross is most
        # reliably derived from explicit Net + spatial deductions.
        # --------------------------------------------------------
        explicit_gross = None
        for row in rows:
            row_text = " ".join(str(w.text) for w in row)
            norm = self._norm(row_text)
            if not any(
                label in norm
                for label in ("gross pay", "gross salary", "gross earnings", "total earnings")
            ):
                continue
            row_money = []
            for w in row:
                parsed = token_money(w)
                if parsed:
                    row_money.append((cx(w), parsed[0], parsed[1]))
            left = [m for m in row_money if m[0] < ded_desc_x]
            if left:
                explicit_gross = max(left, key=lambda m: m[0])[1]
                break

        if explicit_gross:
            result["gross_pay"] = explicit_gross
        else:
            net = self._money_to_float(result["net_pay"])
            deductions = self._money_to_float(result["total_deductions"])
            if net is not None and deductions is not None:
                result["gross_pay"] = self._format_money(net + deductions)

        # --------------------------------------------------------
        # V11 FINAL CROSS-FIELD SANITY
        # --------------------------------------------------------
        # Run only after Gross has been resolved. If the spatial Basic
        # candidate exceeds Gross, the page is internally ambiguous.
        # Returning None is safer than feeding a known-impossible value
        # into verification and creating a false fraud signal.
        basic_value = self._money_to_float(result.get("basic_pay"))
        gross_value = self._money_to_float(result.get("gross_pay"))

        if (
            basic_value is not None
            and gross_value is not None
            and basic_value > gross_value
        ):
            result["basic_pay"] = None

        return result

    def _extract_combined_id_name_from_rows(
        self,
        rows: List[List[Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generic spatial extraction for combined identity rows such as:

            ID / Name : 000205 / Vaibhav Prakash Torase
            Employee No / Name : E123 / Rahul Sharma

        No company names or fixed coordinates are used.
        """
        id_labels = {
            "id", "employee id", "employee no", "employee number",
            "employee code", "emp id", "emp no", "emp number",
            "emp code", "staff id", "staff code", "worker id",
        }

        for row in rows:
            row = sorted(row, key=lambda w: float(w.bbox[0]))
            if not row:
                continue

            row_text = " ".join(str(w.text) for w in row).strip()
            normalized = self._norm(row_text)

            if "name" not in normalized:
                continue

            if not any(label in normalized for label in id_labels):
                continue

            # Prefer a numeric/alphanumeric ID appearing after the label area.
            id_index = None
            employee_id = None

            for index, word in enumerate(row):
                token = str(word.text).strip(" :;|")
                if not self._valid_employee_id(token):
                    continue

                # Avoid obvious year/date tokens.
                if re.fullmatch(r"(?:19|20)\d{2}", token):
                    continue

                id_index = index
                employee_id = self._clean_candidate(token, "employee_id")
                break

            if id_index is None or not employee_id:
                continue

            # Collect a human-name-like sequence after the ID.
            name_tokens: List[str] = []

            for word in row[id_index + 1:]:
                token = str(word.text).strip()

                if token in {"/", ":", ";", "|", "-", "â€“", "â€”"}:
                    if not name_tokens:
                        continue
                    break

                normalized_token = self._norm(token)

                if normalized_token in {
                    "department", "designation", "location", "grade",
                    "pan", "uan", "pf", "bank", "salary",
                }:
                    break

                if re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", token):
                    name_tokens.append(token)
                elif name_tokens:
                    break

            if name_tokens:
                employee_name = self._clean_candidate(
                    " ".join(name_tokens),
                    "name",
                )

                if self._valid_value(employee_name, "name"):
                    return employee_id, employee_name

        return None, None

    # ============================================================
    # LAYOUT ENGINE
    # ============================================================

    def _best_layout_candidate(
        self,
        rows: List[List[Any]],
        field: str,
        aliases: List[str],
    ) -> Optional[str]:
        field_type = self.FIELD_TYPES[field]
        candidates: List[Tuple[float, str]] = []

        for row_index, row in enumerate(rows):
            row = sorted(row, key=lambda w: float(w.bbox[0]))
            normalized_tokens = [self._norm(w.text) for w in row]

            for alias in aliases:
                match = self._find_alias_span(row, normalized_tokens, alias)
                if match is None:
                    continue

                start, end, similarity = match
                label_words = row[start:end + 1]

                lx0 = min(float(w.bbox[0]) for w in label_words)
                lx1 = max(float(w.bbox[2]) for w in label_words)
                ly0 = min(float(w.bbox[1]) for w in label_words)
                ly1 = max(float(w.bbox[3]) for w in label_words)
                lcy = (ly0 + ly1) / 2.0
                label_width = max(1.0, lx1 - lx0)

                # Same-row candidates.
                same_row = row[end + 1:]
                if same_row:
                    for value, distance in self._candidate_groups_right(
                        same_row, lx1, field_type
                    ):
                        score = (
                            similarity * 55.0
                            + max(0.0, 25.0 - distance * 0.10)
                            + self._type_score(value, field_type)
                        )
                        candidates.append((score, value))

                # Below-label candidates.
                for next_index in range(row_index + 1, min(len(rows), row_index + 4)):
                    next_row = sorted(rows[next_index], key=lambda w: float(w.bbox[0]))
                    if not next_row:
                        continue

                    row_y = sum(
                        (float(w.bbox[1]) + float(w.bbox[3])) / 2.0
                        for w in next_row
                    ) / len(next_row)

                    vertical_distance = row_y - lcy
                    if vertical_distance < -2 or vertical_distance > 80:
                        continue

                    for value, x_distance in self._candidate_groups_below(
                        next_row, lx0, lx1, label_width, field_type
                    ):
                        score = (
                            similarity * 55.0
                            + max(0.0, 22.0 - vertical_distance * 0.20)
                            + max(0.0, 10.0 - x_distance * 0.05)
                            + self._type_score(value, field_type)
                        )
                        candidates.append((score, value))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)

        for _score, value in candidates:
            cleaned = self._clean_candidate(value, field_type)
            if self._valid_value(cleaned, field_type):
                return cleaned

        return None

    def _candidate_groups_right(
        self,
        words: List[Any],
        label_x1: float,
        field_type: str,
    ) -> List[Tuple[str, float]]:
        if not words:
            return []

        words = sorted(words, key=lambda w: float(w.bbox[0]))
        groups = self._split_by_large_gap(words)
        output = []

        for group in groups[:3]:
            if not group:
                continue
            value = " ".join(w.text for w in group)
            distance = max(0.0, float(group[0].bbox[0]) - label_x1)

            cleaned = self._clean_candidate(value, field_type)
            if self._valid_value(cleaned, field_type):
                output.append((cleaned, distance))

        return output

    def _candidate_groups_below(
        self,
        row: List[Any],
        label_x0: float,
        label_x1: float,
        label_width: float,
        field_type: str,
    ) -> List[Tuple[str, float]]:
        """
        Dynamic column matching.

        No fixed '+115 px' rule. Search width is derived from label width
        and neighboring word geometry.
        """
        label_center = (label_x0 + label_x1) / 2.0
        margin = max(25.0, label_width * 1.5)

        relevant = []
        for word in row:
            wx0, _, wx1, _ = map(float, word.bbox)
            center = (wx0 + wx1) / 2.0

            if center < label_x0 - margin:
                continue
            if center > label_x1 + margin * 2.5:
                continue

            relevant.append(word)

        if not relevant:
            return []

        groups = self._split_by_large_gap(relevant)
        output = []

        for group in groups:
            value = " ".join(w.text for w in group)
            cleaned = self._clean_candidate(value, field_type)

            if not self._valid_value(cleaned, field_type):
                continue

            group_center = sum(
                (float(w.bbox[0]) + float(w.bbox[2])) / 2.0
                for w in group
            ) / len(group)

            output.append((cleaned, abs(group_center - label_center)))

        return output

    @staticmethod
    def _split_by_large_gap(words: List[Any]) -> List[List[Any]]:
        if not words:
            return []

        words = sorted(words, key=lambda w: float(w.bbox[0]))
        widths = [max(1.0, float(w.bbox[2]) - float(w.bbox[0])) for w in words]
        typical_width = sum(widths) / len(widths)
        gap_limit = max(18.0, typical_width * 1.8)

        groups = [[words[0]]]

        for previous, current in zip(words, words[1:]):
            gap = float(current.bbox[0]) - float(previous.bbox[2])
            if gap > gap_limit:
                groups.append([current])
            else:
                groups[-1].append(current)

        return groups

    def _find_alias_span(
        self,
        row: List[Any],
        normalized_tokens: List[str],
        alias: str,
    ) -> Optional[Tuple[int, int, float]]:
        target = self._norm(alias)
        if not target:
            return None

        best = None

        for start in range(len(row)):
            parts = []
            for end in range(start, min(len(row), start + 6)):
                token = normalized_tokens[end]
                if not token:
                    continue
                parts.append(token)
                combined = " ".join(parts).strip()

                if combined == target:
                    return start, end, 1.0

                similarity = SequenceMatcher(None, combined, target).ratio()

                # Fuzzy matching is deliberately conservative.
                if similarity >= 0.88:
                    if best is None or similarity > best[2]:
                        best = (start, end, similarity)

                if len(combined) > len(target) + 20:
                    break

        return best

    # ============================================================
    # TEXT FALLBACK
    # ============================================================

    def _text_field(self, text: str, field: str) -> Optional[str]:
        field_type = self.FIELD_TYPES[field]

        for line_index, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue

            normalized_line = self._norm(line)

            for alias in self.FIELD_ALIASES[field]:
                normalized_alias = self._norm(alias)

                if normalized_alias not in normalized_line:
                    continue

                pattern = re.compile(
                    re.escape(alias).replace(r"\ ", r"\s+")
                    + r"\s*[:\-]?\s*(.+)$",
                    re.IGNORECASE,
                )
                match = pattern.search(line)

                if match:
                    value = self._clean_candidate(match.group(1), field_type)
                    if self._valid_value(value, field_type):
                        return value

                # Header/value-on-next-line fallback.
                lines = text.splitlines()
                if line_index + 1 < len(lines):
                    value = self._clean_candidate(lines[line_index + 1], field_type)
                    if self._valid_value(value, field_type):
                        return value

        return None

    def _text_money_field(self, text: str, field: str) -> Optional[str]:
        for line_index, line in enumerate(text.splitlines()):
            normalized_line = self._norm(line)

            for alias in self.FIELD_ALIASES[field]:
                normalized_alias = self._norm(alias)

                if normalized_alias not in normalized_line:
                    continue

                pos = normalized_line.find(normalized_alias)
                # Use original line but tolerate approximate normalized offset.
                amounts = self._money_values(line)
                if amounts:
                    # Salary-table rows often contain rate + earned amount.
                    # The final amount is usually the actual/current value.
                    return amounts[-1]

                lines = text.splitlines()
                if line_index + 1 < len(lines):
                    amounts = self._money_values(lines[line_index + 1])
                    if amounts:
                        return amounts[-1]

        return None

    # ============================================================
    # SPECIAL GENERIC FORMS
    # ============================================================

    def _extract_combined_id_name(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r"(?:id|emp(?:loyee)?\s*(?:id|no|number|code))\s*/\s*name"
            r"\s*[:\-]?\s*([A-Za-z0-9\-/]+)\s*/\s*([A-Za-z][A-Za-z .'-]{2,70})",
            r"(?:employee|emp)\s*(?:id|no|number|code)\s*(?:/|&|and)\s*(?:employee\s*)?name"
            r"\s*[:\-]?\s*([A-Za-z0-9\-/]+)\s*(?:/|,|\-)\s*([A-Za-z][A-Za-z .'-]{2,70})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                emp_id = self._clean_candidate(match.group(1), "employee_id")
                name = self._clean_candidate(match.group(2), "name")
                if self._valid_value(emp_id, "employee_id") and self._valid_value(name, "name"):
                    return emp_id, name

        return None, None

    def _extract_pan(self, text: str) -> Optional[str]:
        match = self.PAN_RE.search(text or "")
        return match.group(0).upper() if match else None

    def _extract_period(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Strong salary-specific forms first.
        patterns = [
            r"(?:salary\s*slip|pay\s*slip|payslip)\s*(?:for)?\s*(?:the)?\s*(?:month\s*of)?"
            r"\s*[:\-]?\s*"
            r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)[\s\-/,]*(?:19|20)\d{2})",
            r"(?:pay\s*period|pay\s*month|salary\s*month|salary\s*period|month\s*of|"
            r"salary\s*for|for\s*the\s*month\s*of)\s*[:\-]?\s*"
            r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)[\s\-/,]*(?:19|20)\d{2})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean(match.group(1))

        match = self.MONTH_RE.search(text)
        return self._clean(match.group(0)) if match else None

    # ============================================================
    # VALIDATION / CLEANING
    # ============================================================

    def _clean_candidate(self, value: Any, field_type: str) -> Optional[str]:
        if value is None:
            return None

        value = self._clean(str(value))
        value = re.sub(r"\s+", " ", value).strip(" :|\t-")

        if not value:
            return None

        if field_type == "pan":
            match = self.PAN_RE.search(value)
            return match.group(0).upper() if match else None

        if field_type == "period":
            match = self.MONTH_RE.search(value)
            return self._clean(match.group(0)) if match else None

        if field_type == "money":
            amounts = self._money_values(value)
            return amounts[-1] if amounts else None

        if field_type == "employee_id":
            # Prefer compact alphanumeric IDs, but allow numeric employee numbers.
            tokens = re.findall(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*\b", value)
            for token in tokens:
                if self._valid_employee_id(token):
                    return token
            return None

        if field_type == "account":
            match = re.search(r"\b[0-9Xx*]{6,24}\b", value)
            return match.group(0) if match else None

        # Stop textual fields before another known label begins.
        normalized = value
        for aliases in self.FIELD_ALIASES.values():
            for alias in aliases:
                match = re.search(
                    r"\s+" + re.escape(alias) + r"\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if match:
                    normalized = normalized[:match.start()].strip()
        return normalized or None

    def _valid_value(self, value: Any, field_type: str) -> bool:
        if not value:
            return False

        if field_type == "name":
            return self._valid_employee_name(value)
        if field_type == "employee_id":
            return self._valid_employee_id(value)
        if field_type == "designation":
            return self._valid_designation(value)
        if field_type == "pan":
            return bool(self.PAN_RE.fullmatch(str(value).strip()))
        if field_type == "period":
            return bool(self.MONTH_RE.search(str(value)))
        if field_type == "account":
            return bool(re.fullmatch(r"[0-9Xx*]{6,24}", str(value).strip()))
        if field_type == "money":
            return self._money_to_float(value) is not None
        return False

    def _valid_employee_name(self, value: Any) -> bool:
        if not value:
            return False

        value = str(value).strip()
        if len(value) < 3 or len(value) > 80:
            return False

        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,79}", value):
            return False

        normalized = self._norm(value)
        if normalized in self.BLOCKED_TEXT:
            return False

        blocked_fragments = (
            "employee id", "employee number", "employee code", "payment mode",
            "pan number", "pf number", "bank account", "total earnings",
            "total deductions", "salary slip", "salary details",
        )
        if any(fragment in normalized for fragment in blocked_fragments):
            return False

        # A person's name should normally contain at least two alphabetic tokens.
        alpha_tokens = re.findall(r"[A-Za-z]+", value)
        return len(alpha_tokens) >= 2

    def _valid_employee_id(self, value: Any) -> bool:
        if not value:
            return False

        value = str(value).strip()
        if len(value) < 2 or len(value) > 30:
            return False

        if self._norm(value) in self.BLOCKED_TEXT:
            return False

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/]{1,29}", value):
            return False

        # IDs can be numeric (000205) or alphanumeric (BE1135).
        return bool(re.search(r"\d", value))

    def _valid_designation(self, value: Any) -> bool:
        if not value:
            return False

        value = str(value).strip()
        if len(value) < 2 or len(value) > 80:
            return False

        if not re.search(r"[A-Za-z]", value):
            return False

        normalized = self._norm(value)
        if normalized in self.BLOCKED_TEXT:
            return False

        blocked_fragments = (
            "payment mode", "employee number", "employee id", "pan number",
            "pf number", "bank account", "salary bank", "total earnings",
            "total deductions", "date of joining", "location", "company name",
            "employer name", "salary slip", "pay slip", "payslip",
        )
        if any(fragment in normalized for fragment in blocked_fragments):
            return False

        company_markers = (
            " private limited", " pvt ltd", " limited", " ltd",
            " systems", " technologies", " technology", " solutions",
            " services", " industries", " corporation", " company",
        )
        if any(marker in f" {normalized}" for marker in company_markers):
            return False

        return True

    def _sanitize_fields(self, fields: Dict[str, Any]) -> None:
        validators = {
            "employee_name": "name",
            "employee_id": "employee_id",
            "designation": "designation",
            "pay_period": "period",
            "pan": "pan",
            "bank_account": "account",
            "gross_pay": "money",
            "basic_pay": "money",
            "net_pay": "money",
            "total_deductions": "money",
        }

        for field, field_type in validators.items():
            value = fields.get(field)
            if value is not None:
                value = self._clean_candidate(value, field_type)
            fields[field] = value if self._valid_value(value, field_type) else None

    # ============================================================
    # MONEY / ARITHMETIC
    # ============================================================

    def _money_values(self, value: str) -> List[str]:
        if not value:
            return []

        output = []
        for match in self.MONEY_RE.finditer(value):
            raw = match.group(1).strip()

            # OCR correction: 1.800.00 -> 1,800.00
            if re.fullmatch(r"\d{1,3}\.\d{3}\.\d{2}", raw):
                first_dot = raw.find(".")
                raw = raw[:first_dot] + "," + raw[first_dot + 1:]

            numeric = self._money_to_float(raw)
            if numeric is None:
                continue

            # Avoid dates, tiny percentages, page numbers, etc.
            if numeric < 1:
                continue

            output.append(raw)

        return output

    def _infer_money_fields(self, fields: Dict[str, Any]) -> None:
        """
        Arithmetic is used only to fill a missing value, never to overwrite
        an extracted value. This keeps extraction separate from verification.
        """
        gross = self._money_to_float(fields.get("gross_pay"))
        deductions = self._money_to_float(fields.get("total_deductions"))
        net = self._money_to_float(fields.get("net_pay"))

        if gross is not None and deductions is not None and net is None:
            fields["net_pay"] = self._format_money(gross - deductions)

        elif gross is not None and net is not None and deductions is None:
            inferred = gross - net
            if inferred >= 0:
                fields["total_deductions"] = self._format_money(inferred)

        elif net is not None and deductions is not None and gross is None:
            fields["gross_pay"] = self._format_money(net + deductions)

    @staticmethod
    def _format_money(value: float) -> str:
        return f"{value:.2f}"

    @staticmethod
    def _money_to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            cleaned = (
                str(value)
                .replace(",", "")
                .replace("â‚¹", "")
                .replace("INR", "")
                .replace("Rs.", "")
                .replace("Rs", "")
                .strip()
            )
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_numeric_summary(fields: Dict[str, Any]) -> Dict[str, float]:
        output: Dict[str, float] = {}

        for key in ("gross_pay", "total_deductions", "net_pay", "basic_pay"):
            raw = fields.get(key)
            if raw is None:
                continue
            try:
                output[key] = float(
                    str(raw)
                    .replace(",", "")
                    .replace("â‚¹", "")
                    .replace("INR", "")
                    .replace("Rs.", "")
                    .replace("Rs", "")
                    .strip()
                )
            except (ValueError, TypeError):
                continue

        return output

    # ============================================================
    # ROW / NORMALIZATION HELPERS
    # ============================================================

    @staticmethod
    def _group_words_into_rows(words, y_tolerance: Optional[float] = None):
        if not words:
            return []

        heights = [
            max(1.0, float(word.bbox[3]) - float(word.bbox[1]))
            for word in words
        ]
        typical_height = sorted(heights)[len(heights) // 2]
        tolerance = y_tolerance if y_tolerance is not None else max(3.0, typical_height * 0.45)

        sorted_words = sorted(
            words,
            key=lambda word: (
                (float(word.bbox[1]) + float(word.bbox[3])) / 2.0,
                float(word.bbox[0]),
            ),
        )

        rows: List[List[Any]] = []
        centers: List[float] = []

        for word in sorted_words:
            center_y = (float(word.bbox[1]) + float(word.bbox[3])) / 2.0

            best_index = None
            best_distance = None

            for index, existing_center in enumerate(centers):
                distance = abs(center_y - existing_center)
                if distance <= tolerance and (
                    best_distance is None or distance < best_distance
                ):
                    best_index = index
                    best_distance = distance

            if best_index is None:
                rows.append([word])
                centers.append(center_y)
            else:
                rows[best_index].append(word)
                centers[best_index] = sum(
                    (float(item.bbox[1]) + float(item.bbox[3])) / 2.0
                    for item in rows[best_index]
                ) / len(rows[best_index])

        combined = sorted(zip(centers, rows), key=lambda item: item[0])
        return [
            sorted(row, key=lambda word: float(word.bbox[0]))
            for _, row in combined
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in text.split("\n")
        )

    @staticmethod
    def _norm(value: Any) -> str:
        if value is None:
            return ""

        value = str(value).lower()
        value = value.replace("&", " and ")
        value = value.replace("a/c", "account")
        value = re.sub(r"[^a-z0-9 ]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _type_score(value: str, field_type: str) -> float:
        if not value:
            return 0.0

        if field_type in {"pan", "employee_id", "account", "money", "period"}:
            return 15.0
        if field_type in {"name", "designation"}:
            return 12.0
        return 5.0

    @staticmethod
    def _empty_fields() -> Dict[str, Any]:
        return {
            "employee_name": None,
            "employee_id": None,
            "designation": None,
            "pay_period": None,
            "pan": None,
            "bank_account": None,
            "gross_pay": None,
            "basic_pay": None,
            "net_pay": None,
            "total_deductions": None,
        }
