// Renders the block model written by make_report.py (report_docx.json) into an A4 .docx.
//   node analyze/report_docx.js report_docx.json out.docx
const fs = require("fs");
const {
  AlignmentType, BorderStyle, Document, Header, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak,
  PageNumber, Paragraph, ShadingType, Table, TableCell, TableLayoutType, TableOfContents, TableRow, TextRun,
  VerticalAlign, WidthType, HeightRule,
} = require("docx");

const [modelPath, outPath] = process.argv.slice(2);
const { blocks } = JSON.parse(fs.readFileSync(modelPath, "utf8"));

const mm = (v) => Math.round(v * 56.6929);
const MARGIN = { top: mm(24), right: mm(20), bottom: mm(22), left: mm(22) };
const CONTENT = mm(210) - MARGIN.left - MARGIN.right; // DXA
const PX = (CONTENT / 1440) * 96; // content width in px for ImageRun
const TH = "Leelawadee UI";
const font = (name) => ({ ascii: name, hAnsi: name, cs: TH, eastAsia: TH });

// Thai inside monospace runs: Word shrinks it, so Thai segments fall back to the body font.
const THAI = /([฀-๿][฀-๿\s]*[฀-๿]|[฀-๿])/;

function run(r, extra = {}) {
  if (r.br) return [new TextRun({ break: 1 })];
  const mono = r.code || extra.mono;
  const size = extra.size || (r.code ? 20 : undefined);
  const parts = mono ? r.text.split(THAI).filter((p) => p !== "") : [r.text];
  return parts.map((text) => {
    const thai = mono && THAI.test(text);
    const s = thai ? (size || 22) + 1 : size;
    return new TextRun({
      text,
      font: font(mono && !thai ? "Consolas" : TH),
      bold: r.b || extra.bold, boldComplexScript: r.b || extra.bold,
      subScript: r.sub,
      color: extra.color,
      language: { value: "en-US", bidirectional: "th-TH" }, // lets Word's Thai word breaker wrap lines
      ...(s ? { size: s, sizeComplexScript: s } : {}),
    });
  });
}
const runs = (list, extra) => (list || []).flatMap((r) => run(r, extra));

const PARA = {
  title: { align: AlignmentType.CENTER, size: 42, bold: true, before: 1800, after: 360, line: 420 },
  sub: { align: AlignmentType.CENTER, size: 26, after: 480 },
  abstract: { align: AlignmentType.CENTER, size: 21, after: 900, indent: 700 },
  by: { align: AlignmentType.CENTER, size: 26, bold: true, after: 80 },
  foot: { align: AlignmentType.CENTER, before: 720, after: 520 },
  foot2: { align: AlignmentType.CENTER },
  cap: { size: 20, before: 160, after: 60, keepNext: true },
  figcap: { align: AlignmentType.CENTER, size: 20, after: 200 },
  "toc-title": { size: 32, bold: true, after: 200 },
};

function paragraph(b) {
  const s = PARA[b.style] || {};
  return new Paragraph({
    alignment: s.align || AlignmentType.LEFT,
    keepNext: s.keepNext,
    indent: s.indent ? { left: s.indent, right: s.indent } : undefined,
    spacing: { before: s.before || 0, after: s.after === undefined ? 120 : s.after, line: s.line },
    children: runs(b.runs, { size: s.size, bold: s.bold }),
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "9A9A9A" };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const allBorders = (b) => ({ top: b, bottom: b, left: b, right: b });

// split a cell's runs at <br> into separate paragraphs so alignment applies per line
function cellParagraphs(list, opts) {
  const lines = [[]];
  for (const r of list) (r.br ? lines.push([]) : lines[lines.length - 1].push(r));
  return lines.map((l) => new Paragraph({
    alignment: opts.align, spacing: { before: 0, after: 0, line: 260 },
    children: runs(l, { size: opts.size, mono: opts.mono, bold: opts.bold, color: opts.color }),
  }));
}

function table(b) {
  if (b.style === "members") {
    const widths = [3400, 2800];
    return new Table({
      alignment: AlignmentType.CENTER, width: { size: 6200, type: WidthType.DXA }, columnWidths: widths,
      borders: { ...allBorders(noBorder), insideHorizontal: noBorder, insideVertical: noBorder },
      rows: b.rows.map((r) => new TableRow({
        children: r.map((c, i) => new TableCell({
          width: { size: widths[i], type: WidthType.DXA }, borders: allBorders(noBorder),
          children: cellParagraphs(c.runs, {}),
        })),
      })),
    });
  }
  if (b.style === "grid") {
    const axis = 500, box = 1000, widths = [axis, box, box, box, box];
    return new Table({
      alignment: AlignmentType.CENTER, layout: TableLayoutType.FIXED,
      width: { size: widths.reduce((a, v) => a + v, 0), type: WidthType.DXA }, columnWidths: widths,
      rows: b.rows.map((r, ri) => new TableRow({
        height: { value: ri === b.rows.length - 1 ? 360 : box, rule: HeightRule.EXACT },
        children: r.map((c, i) => {
          const isAxis = c.head;
          const dark = c.cls === "g-x", light = c.cls === "g-sg";
          return new TableCell({
            width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
            borders: allBorders(isAxis ? noBorder : { style: BorderStyle.SINGLE, size: 6, color: "333333" }),
            shading: dark || light ? { type: ShadingType.CLEAR, color: "auto", fill: dark ? "5F5F5F" : "D9D9D9" } : undefined,
            children: cellParagraphs(c.runs, { align: AlignmentType.CENTER, color: isAxis ? "555555" : dark ? "FFFFFF" : undefined,
                                              bold: dark }),
          });
        }),
      })),
    });
  }
  const total = CONTENT;
  const widths = b.widths.map((p) => Math.floor((p / 100) * total));
  widths[widths.length - 1] += total - widths.reduce((a, v) => a + v, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: b.rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0, cantSplit: true,
      children: r.map((c, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA }, borders: allBorders(border),
        verticalAlign: VerticalAlign.CENTER,
        margins: { top: 50, bottom: 50, left: 100, right: 100 },
        shading: c.head ? { type: ShadingType.CLEAR, color: "auto", fill: "E8E8E8" } : undefined,
        children: cellParagraphs(c.runs, {
          size: 20, bold: c.head,
          align: c.cls === "num" ? AlignmentType.RIGHT : AlignmentType.LEFT, mono: c.cls === "num",
        }),
      })),
    })),
  });
}

let listInstance = 0;
const children = [];
for (const b of blocks) {
  switch (b.t) {
    case "pagebreak":
      children.push(new Paragraph({ children: [new PageBreak()], spacing: { after: 0 } }));
      break;
    case "h":
      children.push(new Paragraph({ heading: b.level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
                                    children: runs(b.runs) }));
      break;
    case "p":
      children.push(paragraph(b));
      break;
    case "ol":
    case "ul":
      listInstance += 1;
      for (const item of b.items) {
        children.push(new Paragraph({
          numbering: { reference: b.t === "ol" ? "decimal" : "bullet", level: 0, instance: listInstance },
          spacing: { after: 60 }, children: runs(item),
        }));
      }
      break;
    case "pre":
      b.lines.forEach((line, i) => children.push(new Paragraph({
        indent: { left: 300 }, keepLines: true, keepNext: i < b.lines.length - 1,
        spacing: { before: i === 0 ? 60 : 0, after: i === b.lines.length - 1 ? 140 : 0, line: 276 },
        children: line.length ? runs(line, { mono: true, size: b.style === "eq" ? 21 : 20 }) : [new TextRun("")],
      })));
      break;
    case "img": {
      const w = Math.round(PX * b.pct), h = Math.round((w * b.h) / b.w);
      children.push(new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
        children: [new ImageRun({ type: "png", data: fs.readFileSync(b.src), transformation: { width: w, height: h },
                                  altText: { title: "figure", description: "figure", name: "figure" } })],
      }));
      break;
    }
    case "table":
      children.push(table(b));
      if (b.style !== "grid") children.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      break;
    case "toc":
      children.push(new TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }));
      break;
    default:
      break; // cover_start / cover_end only group blocks
  }
}

const heading = (id, name, size, before, after) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { font: font(TH), size, sizeComplexScript: size, bold: true, boldComplexScript: true, color: "000000" },
  paragraph: { spacing: { before, after }, keepNext: true, outlineLevel: id === "Heading1" ? 0 : 1 },
});
const tocStyle = (id, name, left) => ({
  id, name, basedOn: "Normal", next: "Normal",
  run: { font: font(TH), size: 21, sizeComplexScript: 21 },
  paragraph: { indent: { left }, spacing: { after: 20 } },
});

const doc = new Document({
  creator: "make_report.py",
  title: "BFS vs A* on DJI RoboMaster EP",
  styles: {
    default: {
      document: {
        run: { font: font(TH), size: 22, sizeComplexScript: 22 },
        paragraph: { spacing: { line: 300 } },
      },
    },
    paragraphStyles: [
      heading("Heading1", "Heading 1", 32, 360, 120),
      heading("Heading2", "Heading 2", 26, 240, 80),
      tocStyle("TOC1", "toc 1", 0),
      tocStyle("TOC2", "toc 2", 320),
    ],
  },
  numbering: {
    config: [
      { reference: "decimal", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 300 } } } }] },
      { reference: "bullet", levels: [{ level: 0, format: LevelFormat.BULLET, text: "●", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 300 } }, run: { size: 16 } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: mm(210), height: mm(297) }, margin: { ...MARGIN, header: mm(12) } } },
    headers: {
      default: new Header({
        children: [new Paragraph({ children: [
          new TextRun({ text: "หน้า ", size: 20, sizeComplexScript: 20 }),
          new TextRun({ children: [PageNumber.CURRENT], size: 20, sizeComplexScript: 20 }),
        ] })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => fs.writeFileSync(outPath, buf));
