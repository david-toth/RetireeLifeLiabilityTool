from retiree_life_pricer.soa import parse_improvement_export, parse_mortality_export


def test_parse_soa_mortality_export():
    text = """Table Name:,Example
Row\\Column,1
65,0.01
66,0.02
"""
    parsed = parse_mortality_export(text, "M")

    assert list(parsed.columns) == ["sex", "age", "qx"]
    assert parsed.loc[0, "sex"] == "M"
    assert parsed.loc[1, "qx"] == 0.02


def test_parse_soa_improvement_export():
    text = """Table Name:,Example
Row\\Column,2026,2027
65,0.01,0.02
66,0.03,0.04
"""
    parsed = parse_improvement_export(text, "F")

    assert set(parsed.columns) == {"sex", "age", "year", "improvement"}
    assert len(parsed) == 4
    assert parsed.loc[0, "year"] == 2026
