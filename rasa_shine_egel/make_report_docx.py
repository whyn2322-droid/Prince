from pathlib import Path
import zipfile
from datetime import datetime
import xml.sax.saxutils as saxutils

in_path = Path('Project_Report_mn.txt')
out_path = Path('Project_Report_mn.docx')
if not in_path.exists():
    raise SystemExit('Project_Report_mn.txt not found. Paste the Mongolian text into it first.')

lines = in_path.read_text(encoding='utf-8').splitlines()
project = "Rasa Shine Egel"
now = datetime(2026, 1, 29, 0, 0, 0)
created = now.strftime('%Y-%m-%dT%H:%M:%SZ')

w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
run_props = (
    '<w:rPr>'
    '<w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI" w:cs="Segoe UI" />'
    '<w:lang w:val="mn-MN" />'
    '</w:rPr>'
)

def make_paragraph(text):
    if text == "":
        return f"<w:p><w:r>{run_props}<w:t></w:t></w:r></w:p>"
    safe = saxutils.escape(text)
    return f"<w:p><w:r>{run_props}<w:t xml:space=\"preserve\">{safe}</w:t></w:r></w:p>"

body = "\n".join(make_paragraph(p) for p in lines)

document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{w_ns}">
  <w:body>
{body}
    <w:sectPr/>
  </w:body>
</w:document>
"""

content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

core_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{saxutils.escape(project)} ??????</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
"""

app_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""

word_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>
"""

with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml', content_types)
    z.writestr('_rels/.rels', rels)
    z.writestr('docProps/core.xml', core_xml)
    z.writestr('docProps/app.xml', app_xml)
    z.writestr('word/document.xml', document_xml)
    z.writestr('word/_rels/document.xml.rels', word_rels)

print(f'Created {out_path}')
