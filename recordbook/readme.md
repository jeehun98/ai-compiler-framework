6) 실행 방법

recordbook/web/로 이동

index.html 더블클릭으로 열어도 되고,

권장: 터미널에서

python -m http.server 8000

브라우저에서 http://localhost:8000


AICF 프레임워크에 연동되는 서비스가 아닌 별도 독립형 기록부 / 가이드 UI 를 만든다
UX 먼저 구성
핵심 UX : Block 조립 - Compile - Output / History 확인 + 포트 / 연결선 시각화


전체 화면 레이아웃

Left : Block Palette + Run History
Middle : Workspace (nodes + wires) + Pipeline preview
Right : inspector ( 파라미터 편집 ) + Output( 요약 / LoweredOps / Trace )


파일 / 레이어 분리 원칙
상태는 한 곳 ( state.js ) 에만 둔다 
각 UI / 기능은 자기 책임만 가진 모듈로 분리
렌더는 app.js 가 오케스트레이션만 한다.


JS 파일 역할
js / app.js ( 엔트리 / 오케스트레이터 )
앱 부팅, 이벤트 연결 ( New / Save / Load / Compile ), 전체 rerender 흐름 제어
각 모듈의 mount / render 호출 순서 관리
drawWires() 호출 타이밍 관리 
현재 단계의 mockCompile() 구현 + history 적재


js / state.js ( 상태 / 저장소 )
앱 상태 구조 정의
nodes : 워크스페이스 블럭 목록
selectedId : 선택된 블럭
lastOutput : 마지막 compile 결과
history : run history
localStorage 저장 / 로딩
resetState() 제공


js / dom.js ( DOM 핸들 / 탭 )
화면에 필요한 element 들을 getEls() 로 한 번에 수집
Output 탭 동작 초기화 (initTabs())


js / palette.js ( Block Palette )
팔레트 정의
팔레트 렌더 + 클릭 시 노드 추가
노드 생성은 state 에 push 만 하고, 실제 UI 갱신은 콜백으로 위임


js / workspace.js ( Workspace 노드 렌더 + DnD )
nodesLaayer 에 노드 카드 렌더
노드 클릭 / 삭제 / 위아래 이동
Drag & Drop 으로 순서 변경
노드에 포트 DOM 생성
pipeline preview 텍스트 생성


js / wires.js ( 포트 좌표계산 + SVG 와이어 렌더 )
workspace 좌표계 기준으로 포트 중심 좌표 계산
노드 순서 기반 연결선 생성
SVG path 로 와이어 렌더
드래그 중 와이어 흐림 처리
clear / draw 분리
지금은 순서 연결만 지원
나중에
다중 포트
실데이터 기반 연결
포트 클랙 연결 생성
타입 기반 연결 검증


js / inspector.js ( Inspector 폼 )
selectedId 기반으로 선택된 노드의 params 를 폼으로 렌더
폼 변경 시 state 의 params 갱신
변경 이벤트 발생 시 onChange 로 rerender 요청


js / output.js ( Output 패널 렌더 )
state.lastOUtput 을 Summary / LoweredOps / Trace 패널로 렌더


js / history.js ( Run History 렌더 )
state.history 렌더
히스토리 클릭 시 
워크스페이스 스냅샷 복구
lastOutput 복구
selected 초기화
선택 후 rerender 트리거


js / utils.js ( 유틸 )
uid() 생성
clone() 제공


CSS 파일 역할
styles / base.css
테마 변수
기본 리셋, 공통 클래스


styles / layout.css
상단바, 3-pane 레이아웃 그리드
공통 pane 스타일


styles / blocks.css
palette 버튼 스타일
node 카드 스타일
history 카드 스타일
pipeline preview box


styles / wires.css
SVG wires 레이어 위치 / 크기
port 스타일
wire path 스타일


styles / inspector.css
inspector container + form row 스타일


styles / output.css
 output 탭 + 패널 스타일
key-value row 스타일


데이터 흐름(현재 MVP)
Palette 클릭 → state.nodes.push(node) → rerender
Workspace에서 선택/편집/순서 변경 → state 변경 → rerender
Compile 클릭 → mockCompile()이 state.lastOutput, state.history 갱신 → rerender
rerender 과정에서:
workspace 렌더 → ports 생성
그 다음 frame에서 wires가 ports 좌표 읽어서 선 그림
중간 과정 요약(개발 히스토리)
단일 파일(MVP)에서 시작 → 기능 확장(포트/연결선 필요) → 유지보수/확장성 위해 모듈 분리
와이어 렌더는 DOM 기반(포트 위치를 실시간 계산)으로 구현
→ Drag/Scroll/Resize에 안정적으로 대응
localStorage 저장으로 “recordbook” 성격(기록/회상)을 MVP 단계에서 확보
다음 단계(실용성 업그레이드 순서 추천)
포트 다중화: Linear 입력(x/W/b) 표현 + 연결선 다중
연결 규칙: 순서 연결 → “데이터플로우 연결” (간단한 텐서 이름/타입 기반)
Import(폴더 드롭): AICF export 결과를 드래그해서 Run/Trace 표시 (연동 아님)
Bench 탭: kid별 p50/p90 테이블 + 회귀 표시




