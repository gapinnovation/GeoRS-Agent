from georsagent.taxonomy import load_taxonomy, normalize_label, split_camel_case


def test_class_vocabulary_has_19_canonical_labels():
    tax = load_taxonomy()
    assert len(tax.classes) == 19
    assert "railway station" in tax.classes
    assert "football field" in tax.classes


def test_attribute_vocabulary_matches_abzsl_release():
    tax = load_taxonomy()
    assert tax.attribute_count == 38
    assert {g: len(v) for g, v in tax.attributes.items()} == {
        "objects": 19,
        "dominant_colours": 11,
        "geometric_patterns": 8,
    }


def test_normalization_is_case_and_whitespace_only():
    assert normalize_label("  Railway   Station ") == "railway station"
    assert normalize_label("railway_station") == "railway_station"  # no punctuation rewriting
    assert normalize_label("parking lot") != "parking"  # no synonym replacement


def test_camel_case_split_for_dataset_folders():
    assert split_camel_case("railwayStation") == "railway station"
    assert split_camel_case("footballField") == "football field"
    assert split_camel_case("Airport") == "airport"


def test_manuscript_attribute_terms_resolve_as_aliases():
    tax = load_taxonomy()
    assert tax.lookup_attribute("water vehicles").canonical == "water transport"
    assert tax.lookup_attribute("aircraft").canonical == "airplane"
    assert tax.lookup_attribute("Grey").status == "alias"
    assert tax.lookup_attribute("dock").status == "canonical"
    assert tax.lookup_attribute("turquoise").status == "unknown"


def test_original_annotation_columns_are_mapped():
    tax = load_taxonomy()
    assert tax.annotation_columns["Banchina"] == "dock"
    assert tax.annotation_columns["Mezzi di trasporto via acqua"] == "water transport"
    assert len({v for v in tax.annotation_columns.values()}) == 38
