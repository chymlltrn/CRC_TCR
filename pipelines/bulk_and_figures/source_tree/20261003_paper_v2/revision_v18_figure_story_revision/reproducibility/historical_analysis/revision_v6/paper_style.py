"""Shared, print-scale plotting and export contract for the v3 figures."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

MM=1/25.4
COLORS={'n0':'#0072B2','nplus':'#D55E00','unknown':'#999999','normal':'#0072B2','tumor':'#D55E00','baseline':'#4D4D4D','tcr':'#6F4C9B','q1':'#0072B2','q2':'#CC79A7','gray':'#929AA1','ink':'#222222','grid':'#E6E9EC','light':'#F4F6F8'}
PALETTE=COLORS
MODEL_COLORS={'TCR-BERT':'#0072B2','SCEPTR':'#D55E00','CDR3vec':'#009E73','3-mer + SVD':'#CC79A7'}

def apply_style():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'axes.titleweight':'normal','axes.labelcolor':COLORS['ink'],'text.color':COLORS['ink'],'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.55,'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.direction':'out','ytick.direction':'out','lines.linewidth':0.9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'legend.frameon':False,'legend.handlelength':1.7,'legend.borderaxespad':0.2,'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.dpi':150,'savefig.dpi':600,'axes.unicode_minus':True})

def new_figure(height_mm=150,width_mm=180,**kwargs):
    apply_style();return plt.figure(figsize=(width_mm*MM,height_mm*MM),**kwargs)

def panel_label(ax,label,x=-0.16,y=1.05):
    return ax.text(x,y,label,transform=ax.transAxes,fontsize=10,fontweight='bold',ha='left',va='bottom',clip_on=False)

def panel(ax,label,title=None,x=-0.16,y=1.05):
    panel_label(ax,label,x,y)
    if title:ax.set_title(title,loc='left',pad=7)

def polish(ax,grid_axis=None):
    ax.spines[['top','right']].set_visible(False)
    if grid_axis:ax.grid(axis=grid_axis,color=COLORS['grid'],linewidth=.4,zorder=0)
    ax.set_axisbelow(True)

def zero_line(ax,orientation='vertical'):
    if orientation=='vertical':return ax.axvline(0,color='#747A80',lw=.6,ls=(0,(3,3)),zorder=0)
    return ax.axhline(0,color='#747A80',lw=.6,ls=(0,(3,3)),zorder=0)

def export_figure(fig,stem,source_tables=None,dpi=600):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    outside=[];fonts=[]
    undrawn_tick_labels=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo<=tick.get_loc()<=hi:
                    undrawn_tick_labels.update([tick.label1,tick.label2])
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text() or t in undrawn_tick_labels:continue
        bb=t.get_window_extent(renderer);fonts.append(float(t.get_fontsize()))
        if bb.width>0 and bb.height>0 and (bb.x0<bounds.x0-1 or bb.y0<bounds.y0-1 or bb.x1>bounds.x1+1 or bb.y1>bounds.y1+1):outside.append({'text':t.get_text(),'bbox':list(map(float,bb.bounds))})
    files={}
    for ext in ['pdf','svg','png']:
        p=stem.with_suffix('.'+ext);fig.savefig(p,dpi=dpi,facecolor='white');files[ext]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    preview=stem.with_suffix('.preview.png');fig.savefig(preview,dpi=180,facecolor='white')
    gray=stem.with_suffix('.grayscale.png')
    with Image.open(preview) as im:im.convert('L').save(gray)
    receipt={'status':'REQUIRES_ACTUAL_VISUAL_REVIEW','width_mm':round(fig.get_figwidth()/MM,2),'height_mm':round(fig.get_figheight()/MM,2),'raster_dpi':dpi,'font_points_min':min(fonts) if fonts else None,'out_of_canvas_text':outside,'source_tables':[str(x) for x in (source_tables or [])],'files':files,'note':'Export geometry check does not assess scientific validity or replace actual image review.'}
    stem.with_suffix('.export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

apply_style()
