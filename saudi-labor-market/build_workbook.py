import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import os
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
OUT = HERE + "سوق_العمل_السعودي_تحليل_التوطين.xlsx"
FONT, NAVY, LIGHT, GREEN = "Arial", "1F3864", "D9E2F3", "E2EFDA"
thin = Side(style="thin", color="BFBFBF")
B = Border(left=thin, right=thin, top=thin, bottom=thin)

AR = {
    "Agriculture": "الزراعة والحراجة وصيد الأسماك", "Mining": "التعدين واستغلال المحاجر",
    "Manufacturing": "الصناعات التحويلية", "Electricity": "الكهرباء والغاز والتكييف",
    "Water": "المياه والصرف الصحي وإدارة النفايات", "Construction": "التشييد",
    "Trade": "تجارة الجملة والتجزئة وإصلاح المركبات", "Transport": "النقل والتخزين",
    "Accommodation": "الإقامة وخدمات الطعام", "ICT": "المعلومات والاتصالات",
    "Finance": "الأنشطة المالية والتأمين", "RealEstate": "الأنشطة العقارية",
    "Professional": "الأنشطة المهنية والعلمية والتقنية", "Admin": "الخدمات الإدارية وخدمات الدعم",
    "Public": "الإدارة العامة والدفاع والضمان الاجتماعي", "Education": "التعليم",
    "Health": "الصحة البشرية والعمل الاجتماعي", "Arts": "الفنون والترفيه والتسلية",
    "OtherServices": "أنشطة الخدمات الأخرى", "Households": "الأسر المعيشية بصفتها صاحبة عمل",
    "Extraterritorial": "المنظمات والهيئات الدولية", "NotSpecified": "غير محدد",
}
ns = pd.read_csv(HERE + "data/non_saudi_employed_by_activity.csv", index_col=0, na_values="NA")
tt = pd.read_csv(HERE + "data/total_employed_by_activity.csv", index_col=0, na_values="NA")
Q = [c for c in tt.columns if c != "Q1 2021"]  # source's Q1 2021 total column duplicates non-Saudi
acts = [a for a in tt.index if a != "Total"]
QAR = {q: f"الربع {'الأول الثاني الثالث الرابع'.split()[int(q[1]) - 1]} {q[3:]}" for q in Q}
LATEST, FIRST = Q[-1], Q[0]


def F(**k):
    k.setdefault("name", FONT)
    k.setdefault("size", 11)
    return Font(**k)


def hdr(ws, row, vals, col=1, h=34):
    for i, v in enumerate(vals):
        c = ws.cell(row=row, column=col + i, value=v)
        c.font = F(bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor=NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = B
    ws.row_dimensions[row].height = h


def ttl(ws, t, s):
    ws.sheet_view.rightToLeft = True
    ws["A1"] = t
    ws["A1"].font = F(size=16, bold=True, color=NAVY)
    ws["A2"] = s
    ws["A2"].font = F(size=9, italic=True, color="595959")


SRC = "المصدر: الهيئة العامة للإحصاء، إحصاءات سوق العمل المبنية على السجلات الإدارية (عبر مرصد أسواق العمل والهجرة الخليجية GLMM)."
wb = Workbook()
ov = wb.active
ov.title = "نظرة عامة"

# ---- raw tables ------------------------------------------------------------ #
def raw(name, df, label):
    ws = wb.create_sheet(name)
    ttl(ws, label, SRC + " عدد الأفراد المشتغلين.")
    hdr(ws, 4, ["النشاط الاقتصادي"] + [QAR[q] for q in Q])
    for i, a in enumerate(acts + ["Total"], start=5):
        ws.cell(row=i, column=1, value="الإجمالي" if a == "Total" else AR[a])
        for j, q in enumerate(Q, start=2):
            v = df.loc[a, q]
            c = ws.cell(row=i, column=j, value=None if pd.isna(v) else int(v))
            c.number_format = "#,##0"
        for j in range(1, len(Q) + 2):
            c = ws.cell(row=i, column=j)
            c.border = B
            c.font = F(bold=a == "Total", color="0000FF" if (j > 1 and a != "Total") else "000000")
            if a == "Total":
                c.fill = PatternFill("solid", fgColor=LIGHT)
    tot = 5 + len(acts)
    for j in range(2, len(Q) + 2):  # total row as formula so it re-checks the inputs
        L = get_column_letter(j)
        ws.cell(row=tot, column=j, value=f'=IF(COUNT({L}5:{L}{tot - 1})=0,"",SUM({L}5:{L}{tot - 1}))')
    ws.column_dimensions["A"].width = 38
    for j in range(2, len(Q) + 2):
        ws.column_dimensions[get_column_letter(j)].width = 13
    ws.freeze_panes = "B5"
    ws.cell(row=tot + 2, column=1, value="الربع الثالث 2022 غير متاح في المصدر. الأرقام الزرقاء مدخلات من المصدر، والإجمالي معادلة.").font = F(size=9, italic=True)
    return ws, tot


tw, TOT_ROW = raw("إجمالي المشتغلين", tt, "إجمالي المشتغلين حسب النشاط الاقتصادي (سعوديون وغير سعوديين)")
nw, _ = raw("المشتغلون غير السعوديين", ns, "المشتغلون غير السعوديين حسب النشاط الاقتصادي")
T, N = "'إجمالي المشتغلين'", "'المشتغلون غير السعوديين'"
LC = get_column_letter(len(Q) + 1)   # latest quarter column in raw sheets
FC = "B"                             # first quarter column

# ---- التوطين حسب النشاط ---------------------------------------------------- #
sa = wb.create_sheet("التوطين حسب النشاط", 1)
ttl(sa, f"نسبة التوطين حسب النشاط الاقتصادي - {QAR[LATEST]}", SRC)
hdr(sa, 4, ["النشاط الاقتصادي", "إجمالي المشتغلين", "غير السعوديين", "السعوديون", "نسبة التوطين",
            "الحصة من إجمالي التوظيف", f"السعوديون {QAR[FIRST]}", "نمو السعوديين %",
            f"نسبة التوطين {QAR[FIRST]}", "التغير في نسبة التوطين (نقطة مئوية)"], h=46)
for k, a in enumerate(acts):
    r, src = 5 + k, 5 + k
    sa.cell(row=r, column=1, value=AR[a])
    sa.cell(row=r, column=2, value=f"={T}!{LC}{src}")
    sa.cell(row=r, column=3, value=f"={N}!{LC}{src}")
    sa.cell(row=r, column=4, value=f"=B{r}-C{r}")
    sa.cell(row=r, column=5, value=f"=IFERROR(D{r}/B{r},0)")
    sa.cell(row=r, column=6, value=f"=IFERROR(B{r}/$B${5 + len(acts)},0)")
    sa.cell(row=r, column=7, value=f"={T}!{FC}{src}-{N}!{FC}{src}")
    sa.cell(row=r, column=8, value=f'=IFERROR(D{r}/G{r}-1,"")')
    sa.cell(row=r, column=9, value=f"=IFERROR(G{r}/{T}!{FC}{src},0)")
    sa.cell(row=r, column=10, value=f"=(E{r}-I{r})*100")
tr = 5 + len(acts)
sa.cell(row=tr, column=1, value="الإجمالي")
for j, L in ((2, "B"), (3, "C"), (4, "D"), (7, "G")):
    sa.cell(row=tr, column=j, value=f"=SUM({L}5:{L}{tr - 1})")
sa.cell(row=tr, column=5, value=f"=IFERROR(D{tr}/B{tr},0)")
sa.cell(row=tr, column=6, value=f"=SUM(F5:F{tr - 1})")
sa.cell(row=tr, column=8, value=f'=IFERROR(D{tr}/G{tr}-1,"")')
sa.cell(row=tr, column=9, value=f"=IFERROR(G{tr}/{T}!{FC}{TOT_ROW},0)")
sa.cell(row=tr, column=10, value=f"=(E{tr}-I{tr})*100")
fm = {2: "#,##0", 3: "#,##0", 4: "#,##0", 5: "0.0%", 6: "0.0%", 7: "#,##0", 8: "0.0%", 9: "0.0%", 10: "+0.0;-0.0;0.0"}
for r in range(5, tr + 1):
    for j in range(1, 11):
        c = sa.cell(row=r, column=j)
        c.border = B
        c.font = F(bold=r == tr, color="008000" if j in (2, 3) and r < tr else "000000")
        if j in fm:
            c.number_format = fm[j]
        if r == tr:
            c.fill = PatternFill("solid", fgColor=LIGHT)
sa.column_dimensions["A"].width = 38
for j in range(2, 11):
    sa.column_dimensions[get_column_letter(j)].width = 15
sa.freeze_panes = "B5"
sa.cell(row=tr + 2, column=1, value="الأرقام الخضراء مسحوبة من أوراق البيانات، والسوداء معادلات. السعوديون = الإجمالي - غير السعوديين.").font = F(size=9, italic=True)

# sorted helper (top activities by Saudization, excluding tiny/unspecified groups) via LARGE + INDEX/MATCH
ch = BarChart()
ch.type = "bar"
ch.title = f"نسبة التوطين حسب النشاط - {QAR[LATEST]}"
ch.add_data(Reference(sa, min_col=5, min_row=4, max_row=tr - 1), titles_from_data=True)
ch.set_categories(Reference(sa, min_col=1, min_row=5, max_row=tr - 1))
ch.legend = None
ch.x_axis.numFmt = "0%"
ch.height, ch.width = 14, 22
sa.add_chart(ch, f"A{tr + 4}")

# ---- الاتجاه الربعي -------------------------------------------------------- #
qt = wb.create_sheet("الاتجاه الربعي", 2)
ttl(qt, "اتجاه التوظيف والتوطين ربعياً", SRC)
hdr(qt, 4, ["الربع", "إجمالي المشتغلين", "غير السعوديين", "السعوديون", "نسبة التوطين", "نمو السعوديين عن الربع السابق"])
QT = [q for q in Q if q != "Q3 2022"]  # not available in the source
for k, q in enumerate(QT):
    r, L = 5 + k, get_column_letter(2 + Q.index(q))
    qt.cell(row=r, column=1, value=QAR[q])
    qt.cell(row=r, column=2, value=f"={T}!{L}{TOT_ROW}")
    qt.cell(row=r, column=3, value=f"={N}!{L}{TOT_ROW}")
    qt.cell(row=r, column=4, value=f"=B{r}-C{r}")
    qt.cell(row=r, column=5, value=f"=D{r}/B{r}")
    qt.cell(row=r, column=6, value="" if k == 0 else f'=IFERROR(D{r}/D{r - 1}-1,"")')
    for j, f_ in zip(range(1, 7), [None, "#,##0", "#,##0", "#,##0", "0.0%", "0.0%"]):
        c = qt.cell(row=r, column=j)
        c.border = B
        c.font = F()
        if f_:
            c.number_format = f_
qend = 4 + len(QT)
for j, w in enumerate([22, 16, 16, 16, 14, 18], start=1):
    qt.column_dimensions[get_column_letter(j)].width = w
qt.cell(row=qend + 2, column=1, value="الربع الثالث 2022 غير متاح في المصدر فلم يُدرج، لذلك نمو الربع الرابع 2022 محسوب مقارنة بالربع الثاني 2022.").font = F(size=9, italic=True)

lc = LineChart()
lc.title = "السعوديون وغير السعوديين (عدد المشتغلين)"
lc.add_data(Reference(qt, min_col=3, max_col=4, min_row=4, max_row=qend), titles_from_data=True)
lc.set_categories(Reference(qt, min_col=1, min_row=5, max_row=qend))
lc.y_axis.numFmt = "#,##0"
lc.height, lc.width = 9, 18
qt.add_chart(lc, f"H4")
lc2 = LineChart()
lc2.title = "نسبة التوطين"
lc2.add_data(Reference(qt, min_col=5, min_row=4, max_row=qend), titles_from_data=True)
lc2.set_categories(Reference(qt, min_col=1, min_row=5, max_row=qend))
lc2.y_axis.numFmt = "0%"
lc2.legend = None
lc2.height, lc2.width = 9, 18
qt.add_chart(lc2, f"H24")

# ---- نظرة عامة ------------------------------------------------------------- #
ttl(ov, "سوق العمل السعودي: التوظيف والتوطين حسب النشاط الاقتصادي", SRC)
S = "'التوطين حسب النشاط'"
cards = [
    (f"إجمالي المشتغلين ({QAR[LATEST]})", f"={S}!B{tr}", "#,##0"),
    ("السعوديون", f"={S}!D{tr}", "#,##0"),
    ("نسبة التوطين", f"={S}!E{tr}", "0.0%"),
    (f"التغير في نسبة التوطين منذ {QAR[FIRST]} (نقطة مئوية)", f"={S}!J{tr}", "+0.0;-0.0;0.0"),
    (f"نمو السعوديين منذ {QAR[FIRST]}", f"={S}!H{tr}", "0.0%"),
]
for k, (lbl, f_, nf) in enumerate(cards):
    col = 1 + k
    c = ov.cell(row=4, column=col, value=lbl)
    c.font = F(bold=True, color="FFFFFF", size=10)
    c.fill = PatternFill("solid", fgColor=NAVY)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    v = ov.cell(row=5, column=col, value=f_)
    v.font = F(bold=True, size=16, color=NAVY)
    v.fill = PatternFill("solid", fgColor=LIGHT)
    v.alignment = Alignment(horizontal="center", vertical="center")
    v.number_format = nf
    v.border = B
ov.row_dimensions[4].height = 44
ov.row_dimensions[5].height = 34

# Top / bottom 5 by Saudization among activities with >= 50,000 workers
ov["A8"] = "أعلى 5 أنشطة في نسبة التوطين (الأنشطة التي يزيد عدد المشتغلين فيها عن 50,000)"
ov["A8"].font = F(bold=True, size=12, color=NAVY)
ov["A16"] = "أقل 5 أنشطة في نسبة التوطين (نفس الشرط)"
ov["A16"].font = F(bold=True, size=12, color=NAVY)
# helper column on activity sheet: Saudization only for large activities, tiny tie-breaker to keep ranks unique
for r in range(5, tr):
    sa.cell(row=r, column=12, value=f'=IF(AND(B{r}>=50000,A{r}<>"غير محدد"),E{r}+ROW()/10^9,"")')
    sa.cell(row=r, column=12).font = F(size=8, color="A6A6A6")
sa.cell(row=4, column=12, value="مساعد للترتيب").font = F(size=8, color="A6A6A6")
for block, start, fn in ((0, 9, "LARGE"), (1, 17, "SMALL")):
    hdr(ov, start, ["الترتيب", "النشاط الاقتصادي", "نسبة التوطين", "عدد السعوديين", "إجمالي المشتغلين"], h=22)
    for k in range(5):
        r = start + 1 + k
        key = f"{fn}({S}!$L$5:$L${tr - 1},{k + 1})"
        m = f"MATCH({key},{S}!$L$5:$L${tr - 1},0)"
        ov.cell(row=r, column=1, value=k + 1)
        ov.cell(row=r, column=2, value=f"=INDEX({S}!$A$5:$A${tr - 1},{m})")
        ov.cell(row=r, column=3, value=f"=INDEX({S}!$E$5:$E${tr - 1},{m})").number_format = "0.0%"
        ov.cell(row=r, column=4, value=f"=INDEX({S}!$D$5:$D${tr - 1},{m})").number_format = "#,##0"
        ov.cell(row=r, column=5, value=f"=INDEX({S}!$B$5:$B${tr - 1},{m})").number_format = "#,##0"
        for j in range(1, 6):
            c = ov.cell(row=r, column=j)
            c.border = B
            c.font = F()
            c.fill = PatternFill("solid", fgColor=GREEN if block == 0 else "FCE4D6")
for j, w in enumerate([18, 40, 16, 18, 18], start=1):
    ov.column_dimensions[get_column_letter(j)].width = w

# ---- المصدر والملاحظات ----------------------------------------------------- #
nt = wb.create_sheet("المصدر والملاحظات")
nt.sheet_view.rightToLeft = True
nt["A1"] = "المصدر والملاحظات المنهجية"
nt["A1"].font = F(size=16, bold=True, color=NAVY)
notes = [
    ("المصدر الأصلي", "الهيئة العامة للإحصاء (GASTAT)، إحصاءات سوق العمل المبنية على السجلات الإدارية."),
    ("الجداول المستخدمة", "إجمالي المشتغلين حسب النشاط الاقتصادي (الربع الأول 2021 - الربع الأول 2025)، والمشتغلون غير السعوديين حسب النشاط الاقتصادي (الربع الأول 2021 - الربع الثالث 2025)، كما نشرها مرصد GLMM:"),
    ("", "https://gulfmigration.grc.net/saudi-arabia-saudi-employed-population-by-main-group-of-economic-activities-q1-2021-q1-2025-2/"),
    ("", "https://gulfmigration.grc.net/saudi-arabia-non-saudi-employed-population-by-main-group-of-economic-activities-q1-2021-q3-2025/"),
    ("التغطية", "لا تشمل البيانات العاملين في القطاعات العسكرية والأمنية، ولا السعوديين غير المسجلين في الخدمة المدنية أو التأمينات الاجتماعية أو سجلات العمالة المنزلية، ولا غير السعوديين الذين تقل مدة عملهم عن ثلاثة أشهر."),
    ("فترة التحليل", "من الربع الثاني 2021 إلى الربع الأول 2025. استُبعد الربع الأول 2021 لأن عمود الإجمالي في الجدول المنشور مطابق لعمود غير السعوديين (خطأ في المصدر). الربع الثالث 2022 غير متاح في المصدر."),
    ("طريقة الحساب", "السعوديون = إجمالي المشتغلين - غير السعوديين. نسبة التوطين = السعوديون ÷ إجمالي المشتغلين. التغير بالنقاط المئوية = الفرق بين نسبتي التوطين × 100."),
    ("التحقق", "مجموع الأنشطة في كل ربع يساوي الإجمالي المنشور في المصدر تماماً للجدولين."),
    ("تنبيه", "التغيرات الكبيرة في بعض الأنشطة بين الأرباع (مثل الزراعة والتعليم وغير محدد) قد تعكس إعادة تصنيف في السجلات الإدارية وليس تغيراً فعلياً في التوظيف."),
]
for k, (a, b) in enumerate(notes, start=3):
    nt.cell(row=k, column=1, value=a).font = F(bold=True)
    c = nt.cell(row=k, column=2, value=b)
    c.font = F(color="0563C1" if b.startswith("http") else "000000")
    c.alignment = Alignment(wrap_text=True, vertical="top")
    if b.startswith("http"):
        c.hyperlink = b
nt.column_dimensions["A"].width = 20
nt.column_dimensions["B"].width = 110

for ws in wb.worksheets:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
wb.active = 0
wb.save(OUT)
print(wb.sheetnames, "tr", tr, "TOT_ROW", TOT_ROW, "LC", LC)
