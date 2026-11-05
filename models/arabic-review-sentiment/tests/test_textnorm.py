from arsent.textnorm import normalize_arabic


def test_folds_alef_ya_and_ta_marbuta():
    assert normalize_arabic("أحمد إلى آخر مكتبة") == "احمد الي اخر مكتبه"


def test_removes_diacritics_and_elongation():
    assert normalize_arabic("جَمِيـــل") == "جميل"


def test_shortens_repeats_but_keeps_numbers():
    assert normalize_arabic("روووعة 2000!!!!") == "رووعه 2000!!"


def test_converts_arabic_digits_and_drops_urls():
    assert normalize_arabic("قرأته ١٢ مرة http://x.co/a") == "قراته 12 مره"


def test_keeps_negation_words():
    assert "لا" in normalize_arabic("لا أنصح به").split()


def test_is_idempotent():
    once = normalize_arabic("كتــاب رااائع جداً")
    assert normalize_arabic(once) == once
