"""Rebuild lecture figures from the explicitly approved tiny public samples. No network/training."""
from pathlib import Path
import json, math, html, hashlib
import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'lecture/figures'; MEDIA=ROOT/'lecture/media'; OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(ROOT/'lecture/fonts/Pretendard-Regular.otf')
plt.rcParams.update({'font.family':'Pretendard','font.size':12,'axes.titlesize':14,'axes.labelsize':11,
 'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7','axes.edgecolor':'#a9b5c7','axes.labelcolor':'#17263d',
 'text.color':'#17263d','xtick.color':'#50627b','ytick.color':'#50627b','grid.color':'#dbe2eb',
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','savefig.facecolor':'#fbfaf7'})
BLUE='#3157d5';TEAL='#008b86';ORANGE='#dc8b42';PURPLE='#8b65bd';INK='#17263d';BG='#fbfaf7'
FIGURES={}
def save(fig,name,chapter,title,notes,source='SteSpeech 저자 작성',kind='개념도 / 실행 결과 아님'):
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.15)
 plt.close(fig)
 FIGURES[name]={'file':'figures/'+name+'.svg','chapter':chapter,'title':title,'notes':notes,'source':source,'kind':kind}
def stft(x,sr,nfft=1024,hop=256):
 x=np.pad(x,(nfft//2,nfft//2));frames=np.lib.stride_tricks.sliding_window_view(x,nfft)[::hop]
 return np.fft.rfft(frames*signal.windows.hann(nfft,sym=False),axis=-1).T

def melbank(sr,nfft=1024,nmel=80):
 hz=lambda m:700*(10**(m/2595)-1)
 f=hz(np.linspace(0,2595*np.log10(1+sr/2/700),nmel+2));bins=np.linspace(0,sr/2,nfft//2+1)
 return np.maximum(0,np.minimum((bins[None]-f[:-2,None])/(f[1:-1]-f[:-2])[:,None],(f[2:,None]-bins[None])/(f[2:]-f[1:-1])[:,None]))
def resample(x,sr,new):
 g=math.gcd(sr,new);return signal.resample_poly(x,new//g,sr//g)
def load(name,rate=None):
 x,sr=sf.read(MEDIA/(name+'.wav'),always_2d=True);x=x.mean(1)
 if rate and rate!=sr:x=resample(x,sr,rate);sr=rate
 return x,sr

def image(ax,a,duration,sr=None,mel=False,title='',vmin=-70,vmax=0):
 hi=80 if mel else sr/2000
 im=ax.imshow(a,origin='lower',aspect='auto',extent=[0,duration,0,hi],cmap='magma',vmin=vmin,vmax=vmax,interpolation='nearest')
 ax.set(xlabel='시간 (초)',ylabel='Mel filter index' if mel else '주파수 (kHz)',title=title)
 return im

def speech_plate(name,chapter):
 x,sr=load(name,16000 if name=='kspon' else 22050);dur=len(x)/sr
 nfft,hop=(512,160) if name=='kspon' else (1024,256)
 z=abs(stft(x,sr,nfft,hop));mel=melbank(sr,nfft)@z
 db=20*np.log10(np.maximum(z,1e-7)/max(z.max(),1e-7));mdb=20*np.log10(np.maximum(mel,1e-7)/max(mel.max(),1e-7))
 fig,ax=plt.subplots(3,1,figsize=(10,6),layout='constrained',gridspec_kw={'height_ratios':[.75,1,1]})
 t=np.arange(len(x))/sr;ax[0].plot(t[::4],x[::4],c=TEAL,lw=.6);ax[0].set(xlim=(0,dur),ylabel='진폭',title=f'{name.upper()} 공개 예제 한 발화 · {dur:.2f}초 · 분석 {sr/1000:g} kHz')
 im=image(ax[1],db,dur,sr,title='STFT magnitude · 시간에 따른 주파수 성분')
 fig.colorbar(im,ax=ax[1],label='dB / peak',pad=.015,shrink=.85)
 im=image(ax[2],mdb,dur,mel=True,title='같은 발화의 80-bin mel · 에너지를 주파수 구간별로 묶기')
 fig.colorbar(im,ax=ax[2],label='dB / peak',pad=.015,shrink=.85)
 save(fig,name+'_speech',chapter,'실제 발화 하나를 세 가지 표현으로 보기',[
 '세 그림의 가로축은 동일한 실제 발화의 시간입니다. 세로 줄은 음성의 순간 변화, 가로 무늬는 지속되는 주파수 성분을 보여 줍니다.',
 'STFT와 mel은 보기 편하게 각 표현의 최대값 대비 dB로 표시했습니다. 학습 코드는 HTK magnitude mel의 자연로그를 사용합니다.',
 f'n_fft={nfft}, hop={hop}, periodic Hann, center zero padding. 이 그림은 모델 출력이나 인식 성능이 아닙니다.'],
 json.loads((MEDIA/(name+'.json')).read_text())['source'],'실제 공개 음성에서 직접 계산')
 return x,sr,z,mel

def main():
 kss,sr,z,m=speech_plate('kss',10);kspon,ksr,kz,km=speech_plate('kspon',4)
 # Detailed real waveform / periodicity, no guessed transcript alignment.
 energy=signal.convolve(kss*kss,np.ones(1024)/1024,mode='same');center=int(np.argmax(energy));start=max(0,center-500);end=min(len(kss),start+1000)
 fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained')
 t=np.arange(len(kss))/sr;ax[0].plot(t[::3],kss[::3],lw=.7,c=TEAL);ax[0].axvspan(start/sr,end/sr,color=ORANGE,alpha=.35);ax[0].set(xlabel='시간 (초)',ylabel='진폭',title='KSS 실제 파형 · 표시한 약 45 ms를 아래에서 확대')
 ax[1].plot(t[start:end]*1000,kss[start:end],lw=1.1,c=BLUE);ax[1].set(xlabel='시간 (ms)',ylabel='진폭',title='작은 구간에서는 샘플 간 진폭 변화와 반복 주기가 보입니다');ax[1].grid(alpha=.5)
 save(fig,'wave_zoom',1,'발화 전체와 45 ms 확대는 서로 다른 정보를 줍니다',['위에서는 발화의 시작·끝과 에너지 변화를 봅니다. 아래에서는 음성의 짧은 주기 구조를 봅니다.','강한 에너지 구간을 자동 선택했습니다. 사람이 발음 경계를 판정하거나 강제 정렬한 결과가 아닙니다.'],'KSS 공개 예제 · 직접 계산','실제 공개 음성')
 # Time-frequency tradeoff on identical waveform.
 fig,ax=plt.subplots(2,2,figsize=(10,5.5),layout='constrained')
 for a,n in zip(ax.flat,[256,512,1024,2048]):
  q=abs(stft(kss,sr,n,128));db=20*np.log10(np.maximum(q,1e-7)/q.max());im=image(a,db,len(kss)/sr,sr,title=f'window {n/sr*1000:.1f} ms · Δf {sr/n:.1f} Hz')
 fig.colorbar(im,ax=ax,label='dB / peak',shrink=.6,pad=.015)
 save(fig,'resolution',2,'같은 음성에서 window만 바꾸면 무엇이 달라질까요?',['네 패널은 같은 KSS 발화입니다. hop은 128 samples로 고정했습니다.','긴 window는 주파수 성분을 더 좁게 분리하지만 시간 변화가 넓게 섞입니다. hop을 줄이는 것과 다른 변화입니다.'],'KSS 공개 예제 · STFT 직접 계산','실제 음성 / 전처리 비교')
 # Mel filter bank and real frame spectra.
 fig,ax=plt.subplots(2,1,figsize=(10,5.5),layout='constrained');w=melbank(sr,nmel=20);f=np.linspace(0,sr/2,513)
 for bank in w:ax[0].plot(f/1000,bank,lw=1.3)
 ax[0].set(xlabel='주파수 (kHz)',ylabel='필터 가중치',title='그림용 20개 HTK mel 삼각 필터 · 실제 학습은 80개')
 frame=np.argmax(m.sum(0));ax[1].plot(f/1000,20*np.log10(np.maximum(z[:,frame],1e-6)),c=BLUE,lw=.8);ax[1].set(xlabel='주파수 (kHz)',ylabel='20 log₁₀ |X|',title='실제 KSS 한 프레임의 FFT magnitude · 필터별 가중합의 입력')
 save(fig,'mel_filters',2,'Mel은 주파수 축을 단순히 그림으로 늘리는 연산이 아닙니다',['각 삼각 필터가 이웃 FFT bin을 가중합합니다. 낮은 주파수에 더 촘촘한 구간을 둡니다.','mel = W × |X|. 이 강의 구현은 power 대신 magnitude를 합산한 뒤 자연로그를 적용합니다.'],'HTK mel 수식 / KSS 공개 예제','실제 음성 + 해석적 필터')
 # SpecAugment actual input, explicit illustrative fixed mask.
 db=20*np.log10(np.maximum(km,1e-7)/km.max());masked=db.copy();masked[22:30,:]=-75;masked[:,40:60]=-75
 fig,ax=plt.subplots(2,1,figsize=(10,5.5),layout='constrained');image(ax[0],db,len(kspon)/ksr,mel=True,title='KsponSpeech 원 특징');im=image(ax[1],masked,len(kspon)/ksr,mel=True,title='설명용 고정 time mask + frequency mask')
 fig.colorbar(im,ax=ax,label='dB / peak',shrink=.6,pad=.015)
 save(fig,'specaugment',8,'음성 내용을 바꾸지 않고 일부 특징을 가립니다',['세로 띠는 특정 시간 구간, 가로 띠는 특정 주파수 구간의 정보를 제거합니다.','시각화를 위해 가린 위치를 어둡게 표시했습니다. 실제 code/asr.py는 자연로그 특징의 해당 값을 0으로 설정합니다.','dev/test에는 이 증강을 적용하지 않습니다. 원 파형에 잡음을 더하는 연산과 다릅니다.'],'Park et al., SpecAugment (2019) / KsponSpeech 공개 예제','실제 특징 + 설명용 마스크')
 build_diagrams()
 build_math_figures(kss,sr,z,m)
 if (MEDIA/'music_mixture.wav').exists():music_figures()
 (ROOT/'lecture/figures.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2))
 print(json.dumps({'figures':len(FIGURES),'directory':str(OUT)},ensure_ascii=False))

# Remaining figure builders are below. Every measured plot states its provenance.

def diagram(name,chapter,title,nodes,edges,notes,source='SteSpeech 구현 소스를 대조해 직접 작성',size=(1000,560)):
 """Editable SVG with explicit connections; node=(id,x,y,w,h,title,subtitle,color)."""
 W,H=size
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#607795"/></marker></defs>',f'<rect width="{W}" height="{H}" fill="{BG}"/>']
 lookup={v[0]:v for v in nodes}
 if name=='conformer':
  p.append('<rect x="345" y="18" width="330" height="470" rx="18" fill="none" stroke="#9caec6" stroke-width="2" stroke-dasharray="7 6"/>')
 for edge in edges:
  a,b,*rest=edge;A=lookup[a];B=lookup[b];label=rest[0] if rest else '';kind=rest[1] if len(rest)>1 else ''
  ax,ay,aw,ah=A[1:5];bx,by,bw,bh=B[1:5]
  if abs(by-ay)>abs(bx-ax):
   x1,y1=ax+aw/2,ay+ah if by>ay else ay;x2,y2=bx+bw/2,by if by>ay else by+bh
  else:
   x1,y1=ax+aw if bx>ax else ax,ay+ah/2;x2,y2=bx if bx>ax else bx+bw,by+bh/2
  if kind=='skip':
   mid=min(ay,by)-22;d=f'M{x1},{y1} V{mid} H{x2} V{y2}'
  else:d=f'M{x1},{y1} L{x2},{y2}'
  p.append(f'<path d="{d}" fill="none" stroke="#607795" stroke-width="2.3" marker-end="url(#arrow)"/>')
  if label:p.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" text-anchor="middle" font-family="Pretendard" font-size="15" fill="{TEAL}">{html.escape(label)}</text>')
 for _,x,y,w,h,label,sub,color in nodes:
  fill={'b':'#e5ebff','t':'#dcf1ec','o':'#fae9d6','p':'#eee5fa','n':'#eaf0f5'}.get(color,color)
  p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#c1cddd" stroke-width="1.2"/>')
  lines=label.split('\n');start=y+h/2-(len(lines)-1)*13-(12 if sub else 0)
  for j,line in enumerate(lines):p.append(f'<text x="{x+w/2}" y="{start+j*26}" text-anchor="middle" font-family="Pretendard" font-weight="600" font-size="21" fill="{INK}">{html.escape(line)}</text>')
  if sub:p.append(f'<text x="{x+w/2}" y="{start+len(lines)*26+1}" text-anchor="middle" font-family="Pretendard" font-size="15" fill="#50627b">{html.escape(sub)}</text>')
 p.append('</svg>');(OUT/(name+'.svg')).write_text(''.join(p))
 FIGURES[name]={'file':'figures/'+name+'.svg','chapter':chapter,'title':title,'notes':notes,'source':source,'kind':'직접 작성 구조도 / 실제 구현과 대응'}

def build_diagrams():
 def N(id,x,y,label,sub='',color='b',w=230,h=85):return(id,x,y,w,h,label,sub,color)
 diagram('task_map',0,'세 작업은 입력과 정답의 연결이 다릅니다',[
 N('a',20,60,'음성 파형','KsponSpeech','t'),N('b',380,60,'Conformer + CTC','음향 → 토큰'),N('c',740,60,'문장','ASR','o'),
 N('d',20,230,'문장 / 자모','KSS','o'),N('e',380,230,'FastSpeech 2','토큰 → mel'),N('f',740,230,'HiFi-GAN → 파형','TTS','t'),
 N('g',20,400,'음악 mixture','MUSDB18-HQ','p'),N('h',380,400,'4개 magnitude 모델','mask → iSTFT'),N('i',740,400,'4개 stem','vocals/drums/bass/other','t')],
 [('a','b'),('b','c'),('d','e'),('e','f'),('g','h'),('h','i')],['ASR은 파형에서 문장으로, TTS는 문장에서 파형으로 갑니다.','분리의 정답은 원래 곡의 stem입니다. 임의 분류 라벨이나 ASR 전사문이 아닙니다.'])
 diagram('project_flow',0,'파일 하나가 결과까지 가는 실행 경로',[
 N('a',30,50,'원본 파일','지정한 로컬 경로','t'),N('b',380,50,'prepare-*','manifest + features'),N('c',730,50,'Dataset / collate','lengths + masks'),
 N('d',730,240,'모델 forward','shape 계약'),N('e',380,240,'train / evaluate','명시적 호출만','o'),N('f',30,240,'checkpoint','config + vocab + weights','p'),N('g',380,430,'infer','파일 출력'),N('h',730,430,'사용자 결과','text / WAV / JSON','t')],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g'),('g','h')],['그림 제작에 사용한 공개 예제는 lecture/media에만 둡니다. 학습 manifest에 섞지 않습니다.','train과 evaluate는 작성되어 있지만 실행하지 않았습니다.'])
 diagram('subsample',3,'길이는 Conv1d를 통과할 때 함께 갱신합니다',[
 N('a',40,70,'입력 특징','[B, T, 80]','t',270),N('b',365,70,'Conv1d stride 2','L₁ = ceil(L / 2)','b',270),N('c',690,70,'Conv1d stride 2','L₂ = ceil(L₁ / 2)','b',270),
 N('d',690,300,'Conformer mask','arange(T₂) < L₂','o',270),N('e',365,300,'CTC input_lengths','L₂와 target length 비교','p',270),N('f',40,300,'예: 101 → 51 → 26','패딩 너비와 유효 길이 구별','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['텐서의 padded T와 발화별 유효 길이를 분리합니다.','0 패딩도 convolution bias·attention을 지나면 0이 아닐 수 있어 중간 mask가 필요합니다.'])
 diagram('transcription',4,'전사 규칙을 고르면 vocabulary와 평가 규칙도 고정합니다',[
 N('a',30,70,'원 전사','(철자)/(발음) + 태그','t',270),N('b',365,70,'규칙 선택','orthographic / phonetic','o',270),N('c',700,70,'정규화된 문장','같은 규칙을 모든 split에','b',270),
 N('d',700,320,'train에서만 vocab','dev/test로 vocab 확장 금지','p',270),N('e',365,320,'문자 → token id','blank=0 예약','b',270),N('f',30,320,'CER / WER 규약','정답과 예측에 동일 적용','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['발음 전사와 철자 전사를 임의 혼합하면 같은 발화에 서로 다른 정답을 주게 됩니다.','raw 문자열은 보존하되 학습에 쓰는 표현을 manifest에 명확히 기록합니다.'],'KsponSpeech 논문 §3 / code/data.py')
 diagram('conformer',6,'Conformer: 넓은 문맥과 가까운 시간 패턴을 함께 처리합니다',[
 N('a',30,40,'Mel + subsampling','[B,T,80] → [B,T₂,144]','t',285),
 N('b',370,40,'½ Feed-forward','잔차 + dropout','b',270),N('c',370,150,'Multi-head attention','4 heads / padding mask','p',270),N('d',370,260,'Convolution module','local temporal pattern','o',270),N('e',370,370,'½ Feed-forward + LN','동일한 [B,T₂,144]','b',270),
 N('f',710,370,'Linear → CTC logits','[B,T₂,V]','t',265),N('g',35,310,'동일 블록 4개','각 블록에 residual','n',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['실제 code/asr.py의 흐름입니다. 각 submodule의 잔차 덧셈 뒤 유효 길이 mask를 적용합니다.','원 논문은 relative position과 convolution BatchNorm을 사용합니다. 이 구현은 sinusoidal position과 LayerNorm으로 단순화했습니다.'],'Gulati et al. (2020), Fig. 1–2 / code/asr.py')
 diagram('conv_module',6,'Conformer convolution module 내부를 펼쳐 봅니다',[
 N('a',40,50,'LayerNorm','[B,T,D]','b',260),N('b',370,50,'Pointwise Conv 1×1','D → 2D','b',260),N('c',700,50,'GLU gating','2D → D','o',260),
 N('d',700,240,'Depthwise Conv k=15','groups=D','p',260),N('e',370,240,'LayerNorm + SiLU','프레임별 정규화','b',260),N('f',40,240,'Pointwise Conv 1×1','D → D','b',260),N('g',370,430,'Dropout + residual','입력과 같은 shape','t',260)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g')],['Pointwise는 채널을 섞고 depthwise는 각 채널의 시간 패턴을 봅니다.','GLU의 절반은 값, 나머지 절반은 sigmoid gate가 되어 곱해집니다.'],'Conformer Fig. 2를 실제 구현에 맞게 재작성')
 diagram('decoder',7,'Greedy와 prefix beam은 같은 확률을 다르게 모읍니다',[
 N('a',20,220,'frame log-probability','[T₂,V]','t',260),N('b',370,80,'Greedy: frame argmax','각 시점 최고 토큰','b',270),N('c',700,80,'collapse','반복 → blank 제거','o',270),
 N('d',370,350,'Prefix beam','같은 prefix 경로 합산','p',270),N('e',700,350,'상위 beam 유지','blank / nonblank 분리','b',270)],
 [('a','b'),('b','c'),('a','d'),('d','e')],['최고 확률 경로 하나가 가장 확률 높은 문장과 항상 같지는 않습니다.','code/asr.py는 외부 언어 모델을 사용하지 않습니다. beam width는 탐색량과 근사 오차를 바꿉니다.'],'CTC / code/asr.py prefix_beam')
 diagram('train_step',8,'한 training step에서 무엇이 바뀌는가',[
 N('a',30,50,'batch + augmentation','train split only','t',270),N('b',365,50,'forward + CTC loss','길이 조건 확인','b',270),N('c',700,50,'backward','gradient 계산','o',270),
 N('d',700,300,'unscale + grad clip','norm ≤ 설정값','p',270),N('e',365,300,'optimizer.step','AMP skip 여부 확인','b',270),N('f',30,300,'scheduler.step','실제 update 이후','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['이 그림은 code/engine.py의 실행 순서입니다. 이번 작업에서는 backward와 optimizer.step을 호출하지 않았습니다.','학습 loss와 dev loss는 서로 다른 자료에서 계산합니다. test는 선택 기준으로 사용하지 않습니다.'])
 diagram('split_protocol',9,'훈련·선택·최종 평가를 분리해야 숫자가 의미를 가집니다',[
 N('a',40,60,'TRAIN','가중치 / vocab 학습','b',260),N('b',370,60,'DEV','checkpoint 선택','o',260),N('c',700,60,'TEST','고정 모델 최종 보고','t',260),
 N('d',40,290,'증강 가능','통계는 train에서 추정','b',260),N('e',370,290,'학습 모드 OFF','dropout / 증강 OFF','o',260),N('f',700,290,'사후 튜닝 금지','실패 사례와 규약 기록','t',260)],
 [('a','b'),('b','c'),('a','d'),('b','e'),('c','f')],['CER/WER는 reference 정규화 규약에 민감합니다. 공백 제거 여부와 token 단위를 적습니다.','KsponSpeech 공개 그림용 예제는 eval_clean에서 받은 한 발화이며 학습에 사용하지 않습니다.'])
 diagram('jamo',10,'문자를 토큰으로 바꾸는 과정도 모델 설계입니다',[
 N('a',30,80,'원문 숫자 / 기호','예: 20분','o',270),N('b',365,80,'정규화된 읽기','이십 분','t',270),N('c',700,80,'Unicode NFD','음절 → 자모','b',270),
 N('d',700,330,'자모 vocabulary','padding=0','p',270),N('e',365,330,'token id sequence','[L]','b',270),N('f',30,330,'embedding + position','[L,D]','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['KSS의 expanded_script를 사용하고 숫자·영문이 남으면 명시적으로 오류를 냅니다. 자동 발음 규칙을 감으로 만들지 않습니다.','그림의 “20분 → 이십 분”은 정규화 원리 예제이며 실제 발화 정렬 결과가 아닙니다.'],'KSS 데이터 정의 / code/data.py')
 diagram('alignment_contract',11,'Duration은 음성 프레임과 텍스트 토큰을 이어 주는 다리입니다',[
 N('a',35,60,'자모 tokens','[L]','b',270),N('b',365,60,'외부 정렬 결과','동일 자모에 대한 duration','o',270),N('c',695,60,'duration 합','Σ dᵢ = mel frame 수','t',270),
 N('d',35,320,'음소 alignment','자모와 단위가 다름','p',270),N('e',365,320,'명시적 변환 / 검증','token 일치 + 길이 일치','b',270),N('f',695,320,'학습 cache','mel / d / F0 / energy','t',270)],
 [('a','b'),('b','c'),('d','e'),('e','b'),('c','f')],['MFA 음소 경계를 그대로 NFD 자모 duration이라고 저장하면 안 됩니다.','본 구현은 실제 alignment JSONL을 요구합니다. 가짜 균등 duration으로 TTS 학습이 준비되었다고 하지 않습니다.'])
 diagram('fastspeech2',12,'FastSpeech 2 교육용 구현의 전체 tensor 흐름',[
 N('a',20,35,'자모 + position','[B,L,128]','t',260),N('b',365,35,'FFT encoder ×4','attention + Conv1d','b',270),N('c',700,35,'Duration predictor','log(1+d) 예측','o',270),
 N('d',365,205,'Length regulator','[B,L,D] → [B,T,D]','p',270),N('e',20,355,'Pitch / energy','프레임별 예측 + Linear','o',270),N('f',365,390,'FFT decoder ×4','position + mask','b',270),N('g',700,390,'Mel linear','[B,T,80]','t',270)],
 [('a','b'),('b','c'),('b','d'),('c','d'),('d','e'),('e','f'),('d','f'),('f','g')],['학습에서는 정답 duration/pitch/energy를 넣고 각 predictor에도 별도 loss를 줍니다. 추론에서는 predictor의 값을 사용합니다.','논문의 pitch CWT·양자화 임베딩 대신 frame-level log-F0와 연속 Linear projection을 쓴 교육용 변형입니다.'],'Ren et al. (2020), Fig. 1 / code/tts.py')
 diagram('hifigan',13,'HiFi-GAN: 80개 mel 채널에서 샘플 파형으로',[
 N('a',20,50,'Mel [B,80,T]','pre Conv → 128 channels','t',275),N('b',365,50,'Upsample ×8 + MRF','64 channels / 8T','b',275),N('c',700,50,'Upsample ×8 + MRF','32 channels / 64T','b',275),
 N('d',700,280,'Upsample ×2 + MRF','16 channels / 128T','b',275),N('e',365,280,'Upsample ×2 + MRF','8 channels / 256T','b',275),N('f',20,280,'Conv → tanh','[B,1,256T]','t',275)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['총 배율은 8×8×2×2=256으로 mel hop=256과 맞습니다.','각 MRF는 kernel 3·7·11의 residual branch를 평균합니다. 채널 폭은 원 논문보다 작은 교육용 설정입니다.'],'Kong et al. (2020), Fig. 1 / code/vocoder.py')
 diagram('gan',14,'판별기와 생성기는 서로 다른 gradient 경로로 업데이트됩니다',[
 N('a',20,40,'실제 waveform','ground truth','t',270),N('b',365,40,'D(real)','real → 1','b',270),N('c',700,40,'D loss / D step','G 출력 detach','o',270),
 N('d',20,270,'mel → Generator','fake waveform','p',270),N('e',365,270,'D(fake)','fake → 0 (D 학습)','b',270),N('f',700,270,'G loss / G step','D freeze, gradient는 G로','o',270),N('g',365,440,'mel + feature matching','정답 특징과 차이','t',310)],
 [('a','b'),('b','c'),('d','e'),('e','c'),('e','f'),('g','f')],['D step에서는 fake.detach()로 G를 갱신하지 않습니다. G step에서는 D 파라미터만 고정하고 입력에 대한 gradient는 유지합니다.','그림은 구현된 학습 순서를 설명합니다. 실제 학습·loss 곡선 생성은 실행하지 않았습니다.'])
 diagram('tts_eval',15,'Mel 오차 하나가 음성 품질 전체를 말하지 않습니다',[
 N('a',25,60,'Teacher-conditioned','정답 d / pitch / energy','b',285),N('b',360,60,'Free prediction','자기 predictor 사용','p',285),N('c',695,60,'Waveform 청취','학습된 vocoder 필요','t',285),
 N('d',25,320,'정렬된 mel L1','조건부 진단 지표','b',285),N('e',360,320,'길이 / 발음 / 운율','framewise 단순 비교 주의','p',285),N('f',695,320,'MOS + 이해 가능성','평가자 / 문장 / 조건 공개','t',285)],
 [('a','d'),('b','e'),('c','f')],['정답 duration을 준 mel loss는 자유 추론의 발음·길이 오류를 가릴 수 있습니다.','청취 실험을 실행하지 않았으므로 MOS 수치나 청취 품질 점수를 만들지 않습니다.'])
 diagram('separator',18,'음악 분리 모델의 입력·네 head·복원 경로',[
 N('a',20,50,'Stereo wave','[B,2,N]','t',260),N('b',365,50,'Complex STFT','[B,2,1025,T]','b',270),N('c',700,50,'4개 SourceEstimator','vocals / drums / bass / other','p',280),
 N('d',700,285,'Magnitude → ratio mask','4개 power의 합으로 정규화','o',280),N('e',365,285,'Mixture complex × mask','입력 phase 유지','b',270),N('f',20,285,'iSTFT','[B,4,2,N]','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')],['각 stem이 자기 모델을 갖습니다. 출력은 동일 길이의 스테레오 4개입니다.','Open-Unmix의 multichannel Wiener EM을 구현한 것은 아닙니다. 여기서는 power ratio mask와 mixture phase를 사용합니다.'],'Open-Unmix / code/separation.py')
 diagram('bilstm',18,'한 SourceEstimator 안에서는 시간의 양쪽을 봅니다',[
 N('a',25,60,'log1p magnitude','[B,T,2F]','t',270),N('b',365,60,'Linear + LayerNorm','2F → 128 / tanh','b',270),N('c',705,60,'BiLSTM ×3','forward 64 + backward 64','p',270),
 N('d',705,320,'입력 feature concat','128 + 128 = 256','o',270),N('e',365,320,'MLP + softplus','256 → 128 → 2F','b',270),N('f',25,320,'양의 magnitude','[B,2,F,T]','t',270)],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('b','d')],['BiLSTM은 미래 frame을 보므로 이 모델을 그대로 온라인 causal 시스템이라고 부를 수 없습니다.','공식 UMX의 정규화와 masking 구조를 단순화한 교육용 magnitude estimator입니다.'])
 diagram('sep_loss',20,'분리 학습은 네 stem의 magnitude 정답을 함께 비교합니다',[
 N('a',20,65,'동일 crop mixture','[B,2,N]','t',280),N('b',365,65,'4개 estimator','Ŝ magnitude','b',280),N('c',710,65,'평균 L1','Σ |Ŝ − |STFT(s)||','o',270),
 N('d',20,325,'정답 stem 4개','[B,4,2,N]','p',280),N('e',365,325,'같은 STFT 설정','[B,4,2,F,T]','b',280),N('f',710,325,'backward → optimizer','이번에는 미실행','t',270)],
 [('a','b'),('b','c'),('d','e'),('e','c'),('c','f')],['mixture와 stem은 같은 시간 구간을 잘라야 합니다. 서로 다른 crop이면 올바른 정답이 아닙니다.','독립 peak normalization은 원래 gain 관계를 깨므로 하지 않습니다.'])
 diagram('checkpoint',20,'재개하려면 가중치 밖의 학습 상태도 저장해야 합니다',[
 N('a',20,50,'model + config','아키텍처와 파라미터','b',280),N('b',365,50,'optimizer + scheduler','momentum / LR step','p',280),N('c',710,50,'RNG + scaler','무작위 / AMP state','o',270),N('d',365,300,'checkpoint','epoch / vocab / best metric','t',280),N('e',365,450,'다음 epoch 재개','실제 update 대조는 미검증','b',280)],
 [('a','d'),('b','d'),('c','d'),('d','e')],['현재 검증은 새 모델로 checkpoint를 읽어 같은 forward 출력을 얻는 것까지입니다.','재개 후 optimizer update의 동등성 검증은 학습 실행 금지 범위 때문에 하지 않았습니다.'])
 diagram('frontiers',22,'확장 모델은 표현과 시간 문맥을 바꾸는 방향으로 읽습니다',[
 N('a',20,55,'ASR: Conformer','local conv + attention','b',280),N('b',365,55,'TTS: FastSpeech 2','duration / pitch / energy','p',280),N('c',710,55,'MSS: Open-Unmix','spectrogram + BiLSTM','t',270),
 N('d',20,320,'Streaming ASR','causal mask / chunk / cache','b',280),N('e',365,320,'Flow / diffusion TTS','분포 모델링 / iterative 생성','p',280),N('f',710,320,'Hybrid / band models','time+frequency / subbands','t',270)],
 [('a','d'),('b','e'),('c','f')],['새 모델을 붙일 때 입력 계약·loss·정답·복원 경로가 무엇이 달라지는지 먼저 확인합니다.','이 페이지의 확장 모델은 읽기 대상입니다. 저장소에 구현했다고 주장하지 않습니다.'],'Conformer / FastSpeech 2 / Hybrid Demucs / Band-Split RoFormer')

def build_math_figures(kss,sr,z,mel):
 # Sampling / quantization: deliberately analytical, never labelled recorded data.
 t=np.linspace(0,1,1200);s=np.linspace(0,1,25);fig,ax=plt.subplots(2,1,figsize=(10,5.2),layout='constrained')
 ax[0].plot(t,np.sin(2*np.pi*3*t),c=TEAL,label='연속 함수');ax[0].stem(s,np.sin(2*np.pi*3*s),linefmt=BLUE,markerfmt='o',basefmt=' ');ax[0].set(title='Sampling: 시간축에서 값을 고르기',xlabel='시간 (초)',ylabel='진폭')
 y=np.sin(2*np.pi*3*t);q=np.round(y*3)/3;ax[1].plot(t,y,c=TEAL,alpha=.6);ax[1].step(t,q,c=ORANGE);ax[1].set(title='Quantization: 진폭을 유한한 단계로 반올림하기',xlabel='시간 (초)',ylabel='진폭')
 save(fig,'sampling',1,'샘플링과 양자화는 서로 다른 축의 이산화입니다',['위는 시간 간격, 아래는 진폭 단계의 변화입니다. 이해를 위해 아주 적은 양자화 단계를 그렸습니다.','실제 음성 그림과 구분한 해석적 sine 예제입니다. 파일의 bit depth와 sample rate를 혼동하지 않습니다.'],kind='해석적 신호 예제')
 # Padding / attention mask.
 fig,ax=plt.subplots(1,2,figsize=(10,4.8),layout='constrained');valid=np.arange(12)[None,:]<np.array([12,8,5,3])[:,None]
 ax[0].imshow(valid,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax[0].set(title='batch 유효 frame mask',xlabel='시간 index',ylabel='발화 index');ax[0].set_yticks(range(4),['L=12','L=8','L=5','L=3'])
 attn=np.tile(valid[2],(12,1)).astype(float);ax[1].imshow(attn,cmap='Blues',vmin=0,vmax=1);ax[1].axhline(4.5,color=ORANGE,lw=3);ax[1].set(title='L=5의 key padding mask',xlabel='key 위치',ylabel='query 위치')
 save(fig,'masks',3,'패딩을 “빈 문맥”으로만 두면 충분하지 않습니다',['흰 영역은 유효하지 않은 frame입니다. attention에서 padded key를 차단합니다.','query의 padded output은 attention 후 다시 0으로 가립니다. convolution과 loss에도 같은 유효 길이 계약을 적용합니다.'])
 # CTC lattice for target aa; skip repeated-label edge prohibited.
 fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained');labels=['blank','가','blank','가','blank'];T=7
 for t in range(T):
  for j,l in enumerate(labels):
   ax.scatter(t,j,s=330,c='#e3eafd' if l!='blank' else '#dcefe9',edgecolors='#b7c9d5',zorder=3)
   if t<T-1:
    for jj in [j,j+1,j+2]:
     if jj>=5 or (jj==j+2 and (labels[jj]=='blank' or labels[jj]==l)):continue
     ax.annotate('',(t+1,jj),(t,j),arrowprops={'arrowstyle':'->','color':'#b7c3d1','lw':.9},zorder=1)
 path=[0,1,1,2,3,3,4];ax.plot(range(T),path,c=ORANGE,lw=3,zorder=4);ax.set(xlim=(-.5,6.5),ylim=(-.6,4.6),xlabel='frame t',ylabel='확장된 target state',title='target = “가가” · 반복 문자는 blank를 거쳐야 구별됩니다');ax.set_yticks(range(5),labels)
 save(fig,'ctc_lattice',5,'CTC는 정렬 하나를 고르지 않고 가능한 경로 확률을 더합니다',['주황색은 가능한 경로 하나입니다. 같은 “가” state를 연속 방문한 뒤 blank를 거쳐 다음 “가”로 이동합니다.','같은 문자가 반복되면 두 state를 건너뛰는 transition이 금지됩니다. 최소 frame 수 = 문자 수 + 인접 반복 수입니다.'],'Graves et al. (2006) / code/asr.py ctc_forward','정렬 원리 개념도')
 # CTC posterior synthetic but computed probabilities.
 probs=np.full((4,12),.04);path=[0,0,1,1,0,0,2,2,0,3,3,0]
 for i,j in enumerate(path):probs[j,i]=.88
 fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');im=ax.imshow(probs,cmap='magma',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(4),['blank','음','성','AI']);ax.set(xlabel='frame',ylabel='token',title='설명용 posterior · 합이 1인 예제 확률, 모델 예측 아님');fig.colorbar(im,ax=ax,label='확률')
 save(fig,'ctc_posterior',5,'피크 사이에 blank가 많아도 문장은 짧게 복원됩니다',['argmax 경로: blank, blank, 음, 음, blank, blank, 성, 성, blank, AI, AI, blank.','연속 중복을 먼저 합치고 blank를 제거하면 “음 성 AI”가 됩니다.'],'CTC collapse 규칙','설명용 확률 행렬 / 학습 결과 아님')
 # Prefix mass, explicitly toy distribution.
 fig,ax=plt.subplots(1,2,figsize=(10,4.8),layout='constrained');ax[0].bar(['경로 A₁','경로 A₂','경로 B₁'],[.30,.28,.42],color=[BLUE,BLUE,ORANGE]);ax[0].set(title='서로 다른 alignment 경로',ylabel='확률',ylim=(0,.7));ax[1].bar(['문장 A','문장 B'],[.58,.42],color=[BLUE,ORANGE]);ax[1].set(title='같은 문장으로 collapse한 경로의 합',ylabel='확률',ylim=(0,.7))
 save(fig,'beam_mass',7,'가장 좋은 경로와 가장 좋은 문장은 다를 수 있습니다',['경로 B₁ 하나의 확률은 가장 높지만 문장 A에 대응하는 두 경로의 합이 더 큽니다.','이 숫자는 greedy와 prefix 합산의 차이를 설명하는 toy 예제이며 실제 ASR 성능이 아닙니다.'],kind='설명용 확률 예제')
 # edit DP hand-derived.
 ref=list('나는학교');hyp=list('나도학교');d=np.zeros((len(ref)+1,len(hyp)+1),int);d[:,0]=np.arange(len(ref)+1);d[0]=np.arange(len(hyp)+1)
 for i in range(1,5):
  for j in range(1,5):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.imshow(d,cmap='Blues');ax.set_xticks(range(5),['∅']+hyp);ax.set_yticks(range(5),['∅']+ref);ax.set(xlabel='hypothesis',ylabel='reference',title='“나는학교” vs “나도학교” · substitution 1개 / reference 4자')
 for (i,j),v in np.ndenumerate(d):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>2 else INK,fontsize=18)
 save(fig,'edit_distance',9,'CER의 분자는 문자 alignment에서 센 오류입니다',['동일 문자는 대각선 비용 0, substitution은 1, insertion/deletion도 1입니다.','이 toy 예제의 CER은 1/4=25%입니다. 실제 corpus 결과가 아니며 corpus CER은 전체 오류 합/전체 reference 길이로 집계합니다.'],kind='수식 계산 예제 / corpus 평가 아님')
 # Frame F0 and energy: same autocorrelation algorithm as code/audio.py.
 nfft=1024;hop=256;frames=np.lib.stride_tricks.sliding_window_view(np.pad(kss,(512,512)),1024)[::256].copy();frames=(frames-frames.mean(1,keepdims=True))*signal.windows.hann(1024,sym=False)
 ac=np.fft.irfft(abs(np.fft.rfft(frames,n=2048))**2,n=2048)[:,:1024];lo=int(sr/500);hi=min(int(sr/60),1023);lag=ac[:,lo:hi+1].argmax(1)+lo;peak=ac[np.arange(len(ac)),lag];en=np.sqrt(np.mean(z*z,axis=0)+1e-8);voiced=(peak/np.maximum(ac[:,0],1e-8)>.3)&(en>1e-3);f0=np.where(voiced,sr/lag,np.nan);tt=np.arange(len(f0))*hop/sr
 fig,ax=plt.subplots(3,1,figsize=(10,5.7),layout='constrained');ax[0].plot(np.arange(len(kss))/sr,kss,c=TEAL,lw=.5);ax[0].set(ylabel='진폭',title='KSS 공개 발화');ax[1].plot(tt,f0,'.',c=BLUE,ms=2);ax[1].set(ylabel='F0 (Hz)',ylim=(50,520),title='자기상관 baseline 추정 · 빈 구간은 unvoiced 판정');ax[2].plot(tt,en,c=ORANGE);ax[2].set(xlabel='시간 (초)',ylabel='RMS |STFT|',title='동일 frame의 spectral energy')
 save(fig,'pitch_energy',11,'Pitch와 energy는 동일한 frame 좌표 위에 정렬합니다',['F0는 60–500 Hz 범위의 자기상관 피크로 추정했습니다. 음성학적 정답이나 전문가 주석이 아닙니다.','무성음 판정과 octave 오류가 생길 수 있습니다. 모델의 음질 점수로 해석하지 않습니다.'],'KSS 공개 예제 / code/audio.py와 같은 자기상관 규칙','실제 음성에서 측정한 전처리 특징')
 # Length regulator exact values, not purported KSS alignment.
 fig,ax=plt.subplots(2,1,figsize=(10,4.8),layout='constrained');colors=[BLUE,TEAL,ORANGE];dur=[2,1,3]
 for i in range(3):ax[0].add_patch(Rectangle((i,0),.9,1,color=colors[i]));ax[0].text(i+.45,.5,f'h{i+1}\nd={dur[i]}',ha='center',va='center',color='white',fontsize=17)
 k=0
 for i,n in enumerate(dur):
  for j in range(n):ax[1].add_patch(Rectangle((k,0),.9,1,color=colors[i]));ax[1].text(k+.45,.5,f'h{i+1}',ha='center',va='center',color='white',fontsize=18);k+=1
 for a in ax:a.axis('off');a.set_ylim(-.15,1.2)
 ax[0].set_xlim(-.1,6);ax[1].set_xlim(-.1,6);ax[0].set_title('token 축 L=3');ax[1].set_title('frame 축 T=Σd=6 · repeat_interleave')
 save(fig,'regulator',12,'Length regulator는 토큰 표현을 정수 duration만큼 반복합니다',['duration [2,1,3]이면 token 3개가 frame 6개로 확장됩니다.','이 duration은 연산 설명을 위한 값입니다. 실제 KSS alignment는 별도로 필요합니다.'],'FastSpeech / code/tts.py regulate','연산 예제 / 실제 정렬 아님')
 # MPD reshape and MRF dilated receptive positions.
 fig,ax=plt.subplots(1,2,figsize=(10,5),layout='constrained');a=np.arange(24).reshape(8,3);ax[0].imshow(a,cmap='Blues',aspect='auto');ax[0].set(title='MPD period=3: [N] → [N/3,3]',xlabel='주기 내 위치',ylabel='시간 row')
 for (i,j),v in np.ndenumerate(a):ax[0].text(j,i,str(v),ha='center',va='center',fontsize=13,color='white' if v>15 else INK)
 for i,dilation in enumerate([1,3,5]):
  ax[1].scatter(np.arange(-8,9),np.full(17,i),c='#d4dce8',s=25);ax[1].scatter(np.array([-1,0,1])*dilation,np.full(3,i),c=[BLUE,TEAL,ORANGE][i],s=130)
 ax[1].set(yticks=range(3),yticklabels=['dilation 1','dilation 3','dilation 5'],xlabel='중심과의 sample 간격',title='Residual block k=3의 첫 conv');ax[1].set_ylim(-.7,2.7)
 save(fig,'mpd_mrf',13,'판별기는 주기를, MRF는 여러 범위의 패턴을 봅니다',['MPD는 주기별 reshape 후 2D convolution을 적용합니다. period는 2·3·5·7·11입니다.','오른쪽은 k=3의 dilation 위치입니다. 실제 MRF는 k=3·7·11 branch와 residual 연결을 함께 사용합니다.'],'HiFi-GAN Fig. 1 / code/vocoder.py')
 # Mel convention plot uses real input.
 natural=np.log(np.maximum(mel,1e-5));db=20*np.log10(np.maximum(mel,1e-5)/mel.max());fig,ax=plt.subplots(1,2,figsize=(10,5),layout='constrained')
 im=ax[0].imshow(natural,origin='lower',aspect='auto',cmap='magma');fig.colorbar(im,ax=ax[0],shrink=.65);ax[0].set(title='학습: ln(W |STFT|)',xlabel='frame',ylabel='mel bin')
 im=ax[1].imshow(db,origin='lower',aspect='auto',cmap='magma',vmin=-70,vmax=0);fig.colorbar(im,ax=ax[1],shrink=.65);ax[1].set(title='그림: 최대값 대비 dB',xlabel='frame',ylabel='mel bin')
 save(fig,'mel_convention',14,'비슷한 색의 mel 그림이라도 실제 값의 정의가 다를 수 있습니다',['두 패널은 같은 KSS 특징을 서로 다른 수치 규약으로 표현한 것입니다. 색이 비슷해도 값의 범위가 다릅니다.','사전 학습 vocoder와 연결할 때 sample rate·FFT·hop·filter·log 규칙을 모두 맞춰야 합니다.'],'KSS 공개 예제 / code/audio.py LogMel','실제 음성 / 수치 규약 비교')
 # Controlled shift, no trained-model outputs.
 a=np.log(np.maximum(mel,1e-5));shifted=np.roll(a,6,axis=1);fig,ax=plt.subplots(2,1,figsize=(10,5.5),layout='constrained')
 for aa,v,title in zip(ax,[a,shifted],['실제 KSS mel','비교 원리용으로 6 frame 순환 이동한 mel · 생성 음성 아님']):im=aa.imshow(v,origin='lower',aspect='auto',cmap='magma');aa.set(title=title,ylabel='mel bin',xlabel='frame')
 fig.colorbar(im,ax=ax,label='ln mel magnitude',shrink=.6,pad=.015)
 save(fig,'mel_shift',15,'시간 정렬이 다르면 framewise 오차 해석도 달라집니다',['아래는 위 특징을 수학적으로 이동한 설명 예제입니다. TTS 모델의 출력이나 성능 비교가 아닙니다.','자유 생성은 duration 자체가 달라질 수 있어 무조건 같은 frame끼리 뺀 숫자로 음질을 판단하지 않습니다.'],'KSS 공개 예제 / 명시적 시간 이동','실제 특징 + 통제된 변형')
 # Phase cancellation: analytic exact example.
 t=np.linspace(0,.05,1000);x=np.sin(2*np.pi*200*t);y=-x;fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained');ax[0].plot(t*1000,x,c=BLUE,label='s₁');ax[0].plot(t*1000,y,c=ORANGE,label='s₂: phase π');ax[0].legend(loc='upper right');ax[0].set(title='같은 magnitude, 반대 phase',ylabel='진폭');ax[1].plot(t*1000,x+y,c=TEAL);ax[1].set(title='mixture = s₁+s₂ = 0 · 사라진 성분은 mask로 되살릴 수 없음',xlabel='시간 (ms)',ylabel='합',ylim=(-1.1,1.1))
 save(fig,'phase',17,'Magnitude만 보고 섞인 신호를 이해할 수 없는 이유',['동일 주파수의 두 신호가 반대 phase이면 합에서 완전히 상쇄됩니다.','|S₁+S₂|는 보통 |S₁|+|S₂|와 다릅니다. mixture phase 재사용의 한계를 설명하는 극단적인 예입니다.'],kind='해석적 신호 / phase 설명')
 # Chunk windows and normalization.
 n=2500;chunk=1000;stride=700;weights=np.zeros(n);fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained')
 for i,start in enumerate(range(0,n,stride)):
  w=np.maximum(signal.windows.hann(chunk,sym=True),1e-3);m=min(chunk,n-start);weights[start:start+m]+=w[:m];ax[0].plot(np.arange(start,start+m),w[:m],label=f'chunk {i+1}')
  if start+m==n:break
 ax[0].legend(ncol=4,loc='upper right',fontsize=10);ax[0].set(title='양수 바닥값을 둔 Hann window로 겹쳐 더하기',ylabel='window weight');ax[1].plot(weights,c=ORANGE,label='weight sum');ax[1].plot(weights/weights,c=TEAL,label='합을 나눈 뒤');ax[1].legend();ax[1].set(title='최종 출력 = weighted prediction sum / weight sum',xlabel='sample index',ylabel='gain')
 save(fig,'overlap',19,'Chunk 경계의 gain 변화는 weight 합으로 정규화합니다',['끝부분은 짧은 valid 길이만 누적하고 원래 N samples로 반환합니다.','그림은 실제 separate_long의 window 규칙을 계산한 것입니다. 분리 모델을 실행한 결과가 아닙니다.'],'code/separation.py separate_long','window 수치 계산')
 # SI-SDR vector geometry.
 fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained');s=np.array([3.,1.]);e=np.array([2.,3.]);proj=np.dot(e,s)/np.dot(s,s)*s
 for v,color,label in [(s,TEAL,'reference s'),(e,BLUE,'estimate ŝ'),(proj,ORANGE,'projection αs')]:
  ax.arrow(0,0,*v,head_width=.10,length_includes_head=True,color=color,lw=2)
  dx,dy=(-.65,-.32) if label.startswith('projection') else (.1,.2)
  ax.text(v[0]+dx,v[1]+dy,label,color=color,fontsize=13)
 ax.plot([proj[0],e[0]],[proj[1],e[1]],'--',c=PURPLE);ax.text(2.6,2.5,'residual e',color=PURPLE);ax.set(xlim=(-.1,4.5),ylim=(-.2,3.6),xlabel='벡터 좌표 1',ylabel='벡터 좌표 2',title='SI-SDR = 10 log₁₀ (||αs||² / ||ŝ−αs||²)');ax.set_aspect('equal');ax.grid(alpha=.4)
 save(fig,'sisdr',21,'SI-SDR는 reference 방향으로 투영해 scale을 제거합니다',['이 2차원 그림은 긴 waveform 벡터의 투영을 단순화한 것입니다. 실제 음악 성능 점수가 아닙니다.','BSS Eval SDR와 다른 지표입니다. 무음 reference는 분모 정의가 불안정하므로 별도 상태로 보고합니다.'],'Le Roux et al. (2019), SDR—half-baked or well done?','벡터 기하 설명')
 # Band split is a diagram, no computed model results.
 fig,ax=plt.subplots(1,2,figsize=(10,5),layout='constrained');img=np.arange(48).reshape(8,6);ax[0].imshow(img,cmap='Blues',origin='lower',aspect='auto');ax[0].set(title='주파수 축을 band로 묶기',xlabel='time',ylabel='frequency')
 for boundary in [1.5,3.5,5.5]:ax[0].axhline(boundary,color=ORANGE,lw=3)
 ax[1].axis('off')
 for i,(lab,col) in enumerate([('Band embedding',TEAL),('시간 축 문맥',BLUE),('Band 간 문맥',PURPLE),('복소수 mask / 복원',ORANGE)]):
  ax[1].add_patch(Rectangle((.05,.8-i*.23),.9,.16,color=col,alpha=.16));ax[1].text(.5,.88-i*.23,lab,ha='center',va='center',fontsize=15)
 save(fig,'bands',22,'Band 모델은 음악의 주파수 구조를 모듈에 직접 반영합니다',['왼쪽은 설명용 주파수 격자입니다. 실제 mel이나 모델 attention map이 아닙니다.','Band-Split RoFormer를 읽을 때 band 정의·시간/대역 문맥·mask 출력의 연결을 확인합니다.'],'Lu et al. (2023), Band-Split RoPE Transformer','논문 개념 재작성')


def music_figures():
 names=['mixture','vocals','drums','bass','other'];waves=[]
 for name in names:
  x,sr=sf.read(MEDIA/('music_'+name+'.wav'),always_2d=True);waves.append(x)
 duration=len(waves[0])/sr
 fig,ax=plt.subplots(5,2,figsize=(11,7),layout='constrained',gridspec_kw={'width_ratios':[1,1.35]})
 for i,(name,x) in enumerate(zip(names,waves)):
  mono=x.mean(1);t=np.arange(len(mono))/sr;ax[i,0].plot(t[::12],mono[::12],c=[INK,BLUE,ORANGE,PURPLE,TEAL][i],lw=.5);ax[i,0].set(ylabel=name,xlim=(0,duration));q=abs(stft(mono,sr,2048,512));db=20*np.log10(np.maximum(q,1e-6));im=ax[i,1].imshow(db,origin='lower',aspect='auto',extent=[0,duration,0,sr/2000],cmap='magma',vmin=-50,vmax=30);ax[i,1].set_ylim(0,12);ax[i,1].set_ylabel('kHz')
 ax[0,0].set_title('동일 구간의 waveform');ax[0,1].set_title('동일 dB 범위의 STFT');ax[-1,0].set_xlabel('시간 (초)');ax[-1,1].set_xlabel('시간 (초)')
 fig.colorbar(im,ax=ax[:,1],label='20 log₁₀ |X|',shrink=.5,pad=.015)
 save(fig,'music_stems',16,'실제 mixture와 네 stem을 같은 시간축으로 비교합니다',['다섯 신호는 같은 곡의 같은 7초 구간입니다. 신호별 peak normalization을 하지 않았습니다.','공식 SiSEC18 7초 WAV preview입니다. MUSDB18-HQ 전체 데이터의 대역폭이나 분리 모델 출력이라고 표기하지 않습니다.'],'Rafii et al. / SiSEC18-MUS 7s Excerpts · Zenodo 1256064','실제 공개 음악과 정답 stem')
 # Stereo real wave segments.
 x=waves[0];n=min(len(x),int(.08*sr));start=int(2*sr);fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained')
 for i,ch in enumerate(['Left','Right']):ax[i].plot(np.arange(n)/sr*1000,x[start:start+n,i],c=[BLUE,TEAL][i],lw=.8);ax[i].set(title=ch,ylabel='진폭',xlabel='구간 시작부터 시간 (ms)')
 save(fig,'stereo',16,'스테레오의 두 채널을 하나로 합치면 정보가 사라집니다',['같은 mixture의 2.00–2.08초 Left/Right입니다. 채널별 진폭과 시점 구조를 유지합니다.','실제 분리 입력은 [B,2,N]입니다. ASR의 mono 전처리를 음악 분리에 그대로 복사하면 안 됩니다.'],'MUSDB 공식 7초 preview · 직접 계산','실제 공개 음악')
 # oracle masks from true stems, explicitly not prediction.
 mags=np.stack([abs(stft(x.mean(1),sr,2048,512)) for x in waves[1:]]);p=np.maximum(mags*mags,1e-10);mask=p/p.sum(0,keepdims=True)
 fig,ax=plt.subplots(2,2,figsize=(10,5.5),layout='constrained')
 for a,q,name in zip(ax.flat,mask,names[1:]):im=a.imshow(q,origin='lower',aspect='auto',extent=[0,duration,0,sr/2000],cmap='viridis',vmin=0,vmax=1);a.set(title=name,xlabel='시간 (초)',ylabel='kHz',ylim=(0,10))
 fig.colorbar(im,ax=ax,label='ratio mask',shrink=.6,pad=.015)
 save(fig,'oracle_masks',17,'정답 stem으로 만든 power ratio mask를 살펴봅니다',['각 시간·주파수 위치에서 네 mask의 합은 1입니다. 밝은 영역은 해당 stem의 상대적 power가 큰 곳입니다.','정답 stem을 이용한 oracle 그림입니다. 학습되지 않은 모델의 예측 mask나 분리 성능 결과가 아닙니다.'],'MUSDB 공개 preview / Mₛ=|Sₛ|² ÷ Σ|Sⱼ|²','실제 정답으로 계산한 oracle mask')
 # Synchronization plot same time patch, no pretend performance.
 fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');colors=[BLUE,ORANGE,PURPLE,TEAL]
 for i,name in enumerate(names[1:]):ax.broken_barh([(0,6),(5,2)],(i-.25,.5),facecolors=[colors[i],colors[i]],alpha=.65)
 ax.axvline(5,c=INK,ls='--');ax.set(yticks=range(4),yticklabels=names[1:],xlabel='곡의 시간 (초)',title='같은 crop start/end를 모든 stem에 적용',xlim=(0,7));ax.invert_yaxis()
 save(fig,'crop_sync',19,'Crop과 chunk는 모든 채널·stem에서 같은 위치여야 합니다',['그림은 7초 예제에 적용할 수 있는 6초 chunk와 1초 overlap 위치를 설명합니다. 실제 모델 추론 결과는 아닙니다.','각 stem마다 무작위 시작점을 다르게 뽑으면 mixture와 정답의 대응이 깨집니다.'],kind='실제 예제 길이에 대한 인덱스 개념도')
 # Reconstruction identity, measured data residual (NOT model evaluation).
 summed=sum(waves[1:]);res=waves[0]-summed;fig,ax=plt.subplots(2,1,figsize=(10,5),layout='constrained');tt=np.arange(len(res))/sr
 ax[0].plot(tt[::12],waves[0][::12,0],c=BLUE,lw=.5,label='mixture L');ax[0].plot(tt[::12],summed[::12,0],c=ORANGE,lw=.5,alpha=.7,label='sum of stems L');ax[0].legend();ax[0].set(title='정답 파일의 mixture와 stem 합 · 데이터 관계 확인',ylabel='진폭');ax[1].plot(tt[::12],res[::12,0],c=TEAL,lw=.6);ax[1].set(title='mixture − stem sum · 별도 압축/양자화에 따라 0이 아닐 수 있음',xlabel='시간 (초)',ylabel='잔차 진폭')
 save(fig,'mixture_sum',21,'먼저 데이터 관계를 확인하고 모델 점수와 구분합니다',['위의 파형은 정답 파일끼리 비교한 것입니다. 분리 모델을 적용하거나 SDR을 평가한 것이 아닙니다.','예제 release의 저장·압축 차이에 따른 잔차와 모델의 분리 오류를 혼동하지 않습니다.'],'MUSDB 공식 7초 preview · 파일 관계 계산','공개 파일의 관계 / 모델 평가 아님')

if __name__=='__main__':main()
