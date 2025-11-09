import os
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')


class HTMLVisualizer():
    def __init__(self, fn_html):
        self.fn_html = fn_html
        self.header = []
        self.rows = []

    def add_header(self, elements):
        self.header = list(elements)

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, elements):
        self.rows.append(elements)

    def _render_media_block(self, element):
        parts = ['<div class="media-block">']
        for key, val in element.items():
            if key == 'text':
                parts.append(
                    '<span class="cell-text">{}</span>'.format(escape(str(val)))
                )
            elif key == 'image':
                parts.append(
                    '<img src="{src}" alt="{alt}" loading="lazy">'.format(
                        src=escape(val),
                        alt=escape(os.path.basename(val))
                    )
                )
            elif key == 'audio':
                parts.append(
                    '<audio controls preload="none">'
                    '<source src="{src}" type="audio/wav">'
                    'Your browser does not support the audio element.'
                    '</audio>'.format(src=escape(val))
                )
            elif key == 'video':
                parts.append(
                    '<video controls preload="metadata">'
                    '<source src="{src}" type="video/mp4">'
                    'Your browser does not support the video tag.'
                    '</video>'.format(src=escape(val))
                )
            elif key == 'youtube':
                embed_src = ''
                start = None
                end = None
                if isinstance(val, dict):
                    video_id = str(val.get('id', '')).strip()
                    start = val.get('start')
                    end = val.get('end')
                    explicit_url = val.get('url')
                else:
                    video_id = str(val).strip()
                    explicit_url = None
                if explicit_url:
                    embed_src = explicit_url
                elif video_id.startswith('http://') or video_id.startswith('https://'):
                    embed_src = video_id
                elif video_id:
                    query_params = ['rel=0']
                    try:
                        if start is not None:
                            start_param = max(0, int(round(float(start))))
                            query_params.append(f'start={start_param}')
                    except (TypeError, ValueError):
                        pass
                    try:
                        if end is not None:
                            end_param = max(0, int(round(float(end))))
                            if end_param > 0:
                                query_params.append(f'end={end_param}')
                    except (TypeError, ValueError):
                        pass
                    query_str = '&'.join(query_params)
                    embed_src = f'https://www.youtube.com/embed/{video_id}?{query_str}'
                if embed_src:
                    embed_src = escape(embed_src, quote=True)
                    parts.append(
                        '<iframe class="yt-frame" src="{src}" '
                        'title="YouTube video player" frameborder="0" '
                        'loading="lazy" allow="accelerometer; autoplay; clipboard-write; '
                        'encrypted-media; gyroscope; picture-in-picture; web-share" '
                        'allowfullscreen></iframe>'.format(src=embed_src)
                    )
                else:
                    parts.append(
                        '<span class="cell-text">Preview unavailable</span>'
                    )
        parts.append('</div>')
        return ''.join(parts)

    def write_html(self):
        if not self.header and not self.rows:
            return

        html_parts = ['<!DOCTYPE html>',
                      '<html lang="en">',
                      '<head>',
                      '<meta charset="UTF-8">',
                      '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                      '<title>Sound Of Pixels Visualization</title>',
                      '<style>',
                      ':root {',
                      '  --bg-color: #0f172a;',
                      '  --panel-color: rgba(15, 23, 42, 0.88);',
                      '  --border-color: rgba(148, 163, 184, 0.25);',
                      '  --accent-color: #38bdf8;',
                      '  --accent-soft: rgba(56, 189, 248, 0.18);',
                      '  --text-color: #e2e8f0;',
                      '  --muted-text: #94a3b8;',
                      '  --header-bg: rgba(30, 41, 59, 0.9);',
                      '  --media-size: 220px;',
                      '}',
                      '*, *::before, *::after { box-sizing: border-box; }',
                      'body {',
                      '  margin: 0;',
                      '  padding: 32px;',
                      '  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;',
                      '  background: linear-gradient(160deg, #020617 0%, #0f172a 40%, #1e293b 100%);',
                      '  color: var(--text-color);',
                      '}',
                      'a { color: inherit; }',
                      '.page-header {',
                      '  max-width: 1040px;',
                      '  margin: 0 auto 24px auto;',
                      '  padding: 24px 28px;',
                      '  background-color: var(--panel-color);',
                      '  border: 1px solid var(--border-color);',
                      '  border-radius: 18px;',
                      '  box-shadow: 0 22px 45px rgba(2, 6, 23, 0.55);',
                      '  backdrop-filter: blur(18px);',
                      '}',
                      '.page-header h1 {',
                      '  margin: 0;',
                      '  font-size: 28px;',
                      '  letter-spacing: 0.4px;',
                      '}',
                      '.page-header p {',
                      '  margin: 10px 0 18px 0;',
                      '  color: var(--muted-text);',
                      '  line-height: 1.6;',
                      '}',
                      '.controls {',
                      '  display: flex;',
                      '  flex-wrap: wrap;',
                      '  gap: 12px 18px;',
                      '  align-items: flex-end;',
                      '}',
                      '.controls-group {',
                      '  display: flex;',
                      '  flex-direction: column;',
                      '  gap: 6px;',
                      '}',
                      '.controls-group label {',
                      '  font-size: 13px;',
                      '  text-transform: uppercase;',
                      '  letter-spacing: 1.2px;',
                      '  color: var(--muted-text);',
                      '}',
                      '.controls input[type=\"search\"] {',
                      '  padding: 10px 14px;',
                      '  border-radius: 12px;',
                      '  border: 1px solid var(--border-color);',
                      '  background: rgba(15, 23, 42, 0.65);',
                      '  color: var(--text-color);',
                      '  font-size: 15px;',
                      '  min-width: 260px;',
                      '}',
                      '.controls input[type=\"search\"]::placeholder { color: var(--muted-text); }',
                      '.controls input[type=\"range\"] {',
                      '  width: 220px;',
                      '  accent-color: var(--accent-color);',
                      '}',
                      '.table-container {',
                      '  max-width: 1040px;',
                      '  margin: 0 auto;',
                      '  overflow-x: auto;',
                      '  background-color: var(--panel-color);',
                      '  border-radius: 18px;',
                      '  border: 1px solid var(--border-color);',
                      '  box-shadow: 0 28px 65px rgba(2, 6, 23, 0.58);',
                      '  backdrop-filter: blur(18px);',
                      '}',
                      '.vis-table {',
                      '  width: 100%;',
                      '  border-collapse: collapse;',
                      '}', 
                      '.vis-table thead th {',
                      '  position: sticky;',
                      '  top: 0;',
                      '  z-index: 5;',
                      '  padding: 14px 12px;',
                      '  background: var(--header-bg);',
                      '  border-bottom: 1px solid var(--border-color);',
                      '  font-weight: 600;',
                      '  text-transform: uppercase;',
                      '  font-size: 13px;',
                      '  letter-spacing: 1.1px;',
                      '}',
                      '.vis-table tbody tr { transition: background-color 220ms ease; }',
                      '.vis-table tbody tr:nth-child(even) { background: rgba(148, 163, 184, 0.05); }',
                      '.vis-table tbody tr:hover { background: var(--accent-soft); }',
                      '.vis-table td {',
                      '  padding: 18px 14px;',
                      '  border-bottom: 1px solid var(--border-color);',
                      '  border-right: 1px solid var(--border-color);',
                      '  vertical-align: top;',
                      '  text-align: center;',
                      '}',
                      '.vis-table td:last-child, .vis-table th:last-child { border-right: none; }',
                      '.media-block {',
                      '  display: flex;',
                      '  flex-direction: column;',
                      '  gap: 10px;',
                      '  align-items: center;',
                      '}',
                      '.media-block img, .media-block video {',
                      '  max-width: var(--media-size);',
                      '  max-height: var(--media-size);',
                      '  border-radius: 14px;',
                      '  border: 1px solid rgba(148, 163, 184, 0.2);',
                      '  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.45);',
                      '  transition: transform 200ms ease, box-shadow 200ms ease;',
                      '}',
                      '.media-block img:hover, .media-block video:hover {',
                      '  transform: translateY(-4px) scale(1.02);',
                      '  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.55);',
                      '}',
                      '.media-block iframe {',
                      '  width: var(--media-size);',
                      '  height: calc(var(--media-size) * 0.5625);',
                      '  border-radius: 14px;',
                      '  border: 1px solid rgba(148, 163, 184, 0.2);',
                      '  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.45);',
                      '}',
                      '.media-block audio {',
                      '  width: var(--media-size);',
                      '  filter: drop-shadow(0 6px 16px rgba(15, 23, 42, 0.65));',
                      '}',
                      '.cell-text {',
                      '  font-weight: 600;',
                      '  font-size: 14px;',
                      '  letter-spacing: 0.4px;',
                      '  color: var(--text-color);',
                      '}',
                      '@media (max-width: 920px) {',
                      '  body { padding: 18px; }',
                      '  .page-header, .table-container { border-radius: 14px; }',
                      '  .controls input[type=\"range\"] { width: 180px; }',
                      '}',
                      '</style>',
                      '</head>',
                      '<body>',
                      '<header class="page-header">',
                      '<h1>Sound Of Pixels: Separation Visualizer</h1>',
                      '<p>Explore mixture inputs, per-source predictions, masks, and ground truth data. '
                      'Use the filter to quickly focus on a specific duet.</p>',
                      '<div class="controls">',
                      '<div class="controls-group">',
                      '<label for="search">Filter rows</label>',
                      '<input id="search" type="search" placeholder="Start typing an instrument, e.g. cello">',
                      '</div>',
                      '<div class="controls-group">',
                      '<label for="sizeRange">Thumbnail size</label>',
                      '<input id="sizeRange" type="range" min="160" max="360" value="220">',
                      '</div>',
                      '</div>',
                      '</header>',
                      '<div class="table-container">',
                      '<table class="vis-table">']

        if self.header:
            html_parts.append('<thead><tr>')
            for element in self.header:
                html_parts.append('<th>{}</th>'.format(escape(str(element))))
            html_parts.append('</tr></thead>')

        html_parts.append('<tbody>')
        for row in self.rows:
            label = ''
            if row and isinstance(row[0], dict) and 'text' in row[0]:
                label = str(row[0]['text']).lower()
            html_parts.append('<tr data-label="{}">'.format(escape(label)))
            for element in row:
                html_parts.append('<td>{}</td>'.format(self._render_media_block(element)))
            html_parts.append('</tr>')
        html_parts.append('</tbody></table></div>')
        html_parts.append('<script>')
        html_parts.append('const searchInput=document.getElementById("search");')
        html_parts.append('const rows=document.querySelectorAll(".vis-table tbody tr");')
        html_parts.append('searchInput.addEventListener("input",function(){')
        html_parts.append('const query=this.value.toLowerCase();')
        html_parts.append('rows.forEach(row=>{')
        html_parts.append('const label=row.getAttribute("data-label")||row.textContent.toLowerCase();')
        html_parts.append('row.style.display=label.includes(query)? "":"none";')
        html_parts.append('});')
        html_parts.append('});')
        html_parts.append('const slider=document.getElementById("sizeRange");')
        html_parts.append('const root=document.documentElement;')
        html_parts.append('const updateSize=value=>{')
        html_parts.append('root.style.setProperty("--media-size", `${value}px`);')
        html_parts.append('};')
        html_parts.append('if(slider){')
        html_parts.append('updateSize(slider.value);')
        html_parts.append('slider.addEventListener("input",()=>updateSize(slider.value));')
        html_parts.append('}')
        html_parts.append('</script>')
        html_parts.append('</body></html>')

        with open(self.fn_html, 'w') as f:
            f.write('\n'.join(html_parts))
