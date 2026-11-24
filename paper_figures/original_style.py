"""Use the same Helvetica-Regular face actually embedded in the original PDFs.
The original macOS TTC-based renderer embedded Regular even for bold requests.
New Matplotlib can resolve other faces in the TTC, so explicitly select Regular.
No font is redistributed; extraction uses the machine's own font, with fallback.
"""
from pathlib import Path
import tempfile,logging,json
from matplotlib import font_manager
from matplotlib.text import Text
FONT_PATH=None

def regular_font():
    global FONT_PATH
    if FONT_PATH:return FONT_PATH
    try:
        p=Path(font_manager.findfont(font_manager.FontProperties(family='Helvetica',weight='normal'),fallback_to_default=False))
    except ValueError:
        p=Path(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='normal')))
    if p.suffix.lower()=='.ttc':
        from fontTools.ttLib import TTFont
        messages=[]
        class Capture(logging.Handler):
            def emit(self,record):messages.append(record.getMessage())
        logger=logging.getLogger('fontTools');h=Capture();prior=logger.propagate;logger.addHandler(h);logger.propagate=False
        try:
            font=TTFont(str(p),fontNumber=0)
            # Glyph order is preserved; drop optional PostScript glyph-name padding.
            post=font['post'];post.formatType=3.0
            head=font['head']
            if head.created<2082844800:head.created=2082844800
            if head.modified<2082844800:head.modified=2082844800
            dest=Path(tempfile.gettempdir())/'atomagent-helvetica-regular-normalized.ttf'
            font.save(dest);font.close()
        finally:
            logger.removeHandler(h);logger.propagate=prior
        unexpected=[m for m in messages if 'post.stringData' not in m and 'timestamp seems very low' not in m]
        if unexpected:raise RuntimeError('Unexpected font warning: '+repr(unexpected))
        FONT_PATH=str(dest)
    else:FONT_PATH=str(p)
    return FONT_PATH

def apply_original_typography(fig):
    path=regular_font()
    for item in fig.findobj(match=Text):
        item.set_fontproperties(font_manager.FontProperties(fname=path,size=item.get_fontsize()))
