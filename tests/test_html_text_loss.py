from app.corpus.filling_parser import to_blocks


def test_mixed_div_keeps_direct_text_and_nested_paragraph():
    html = "<html><body><div>Direct lead-in sentence.<p>Nested paragraph.</p></div></body></html>"
    _, text = to_blocks(html, ".htm")
    assert "Direct lead-in sentence." in text
    assert "Nested paragraph." in text


def test_loose_body_text_is_kept():
    html = "<html><body>Loose body text<p>Wrapped.</p></body></html>"
    _, text = to_blocks(html, ".htm")
    assert "Loose body text" in text
    assert "Wrapped." in text


def test_tail_text_after_inline_tag_is_kept():
    html = "<html><body><div><span>First part</span> tail text stays.</div></body></html>"
    _, text = to_blocks(html, ".htm")
    assert "First part" in text
    assert "tail text stays." in text


def test_nested_table_does_not_corrupt_outer_table():
    html = ("<html><body><table><tr><th>Year</th><th>Revenue</th></tr>"
            "<tr><td>2024</td><td><table><tr><td>60,922</td></tr></table></td></tr>"
            "</table></body></html>")
    blocks, _ = to_blocks(html, ".htm")
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.header == ("Year", "Revenue")
    # the inner table's row must not become a row of the outer table
    assert ("60,922",) not in table.rows
    # the cell still carries the nested value once
    assert any("60,922" in " ".join(row) for row in table.rows)


def test_no_text_duplicated_when_table_nested():
    html = ("<html><body><table><tr><td><table><tr><td>only once</td></tr></table></td></tr>"
            "</table></body></html>")
    _, text = to_blocks(html, ".htm")
    assert text.count("only once") == 1


def test_comments_and_doctype_are_not_page_text():
    raw = ('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
           "<html><!-- Document created using Wdesk 1 --><body><p>Real text.</p>"
           "<!-- Copyright 2017 Workiva --></body></html>")
    _, text = to_blocks(raw, ".htm")
    assert "Real text." in text
    assert "Wdesk" not in text
    assert "Workiva" not in text
    assert "DOCTYPE" not in text
    assert "W3C" not in text


def test_layout_wrapper_table_unwraps_to_real_table():
    # Word-converted filings center content in a 1-cell wrapper table
    html = ("<html><body><table><tr><td>"
            "<table><tr><th>Years ended</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Total net sales</td><td>391,035</td><td>383,285</td></tr></table>"
            "</td></tr></table></body></html>")
    blocks, text = to_blocks(html, ".htm")
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].header == ("Years ended", "2024", "2023")
    assert ("Total net sales", "391,035", "383,285") in tables[0].rows
    assert text.count("391,035") == 1


def test_data_table_with_text_around_nested_table_is_not_unwrapped():
    # a real table whose cell has text PLUS a nested table keeps its own shape
    html = ("<html><body><table><tr><th>Year</th><th>Detail</th></tr>"
            "<tr><td>2024</td><td>breakdown:<table><tr><td>60,922</td></tr></table></td></tr>"
            "</table></body></html>")
    blocks, _ = to_blocks(html, ".htm")
    tables = [b for b in blocks if b.kind == "table"]
    assert tables[0].header == ("Year", "Detail")
    assert any("breakdown:" in " ".join(row) for row in tables[0].rows)
