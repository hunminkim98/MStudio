# MStudio 리팩토링 및 차세대 플랫폼 전환 계획 (Refactoring & Migration Plan)

본 문서는 현재 Python/Tkinter 기반의 MStudio 코드베이스를 **C#/.NET 8/WPF/Direct3D** 기반의 고성능 데스크톱 애플리케이션으로 전환하기 위한 구체적인 **마이그레이션 및 리팩토링 로드맵**을 제시합니다.

참고 문서: `RFP.md` (기술 세부 요구사항)

---

## 1. 현재 아키텍처 분석 (AS-IS) vs 목표 아키텍처 (TO-BE)

기존 Python 코드의 핵심 모듈을 차세대 아키텍처의 대응 컴포넌트로 매핑하여 기능 누락 없는 전환을 보장합니다.

| 구분 | 현재 (Python/Tkinter) | **목표 (C#/.NET 8/WPF)** | 리팩토링 방향 |
| :--- | :--- | :--- | :--- |
| **진입점** | `app.py` (TRCViewer Class) | `App.xaml` / `MainWindow.xaml` | 단일 거대 클래스를 **MVVM 패턴**으로 분리 (ShellView + MainViewModel). |
| **UI 프레임워크** | CustomTkinter + Matplotlib | **WPF + AvalonDock** (오픈소스) | 고정 레이아웃을 **유동적 도킹 시스템**으로 전면 교체. |
| **데이터 관리** | `core/data_manager.py` (Pandas) | **SessionService** (MemoryMappedFile) | 대용량 데이터 로딩 속도 개선을 위해 Pandas 대신 **Struct Array + Span<T>** 최적화 구조 사용. |
| **렌더링** | `gui/opengl/` (PyOpenGL) | **Direct3D 11 (Vortice.Windows)** | Python 인터프리터 오버헤드 제거 및 DirectX 직접 제어로 퍼포먼스 극대화. |
| **그래프** | `markerPlot.py` (Matplotlib) | **Direct3D Line Renderer** / OxyPlot (오픈소스) | Matplotlib의 느린 반응성을 실시간 60fps 렌더링으로 대체. |
| **재생 제어** | `core/animation_controller.py` | **TimelineService** | 단일 타이머 루프에서 **고정밀 멀티미디어 타이머** 기반 동기화 시스템으로 변경. |
| **알고리즘** | `core/outlier_detector.py` | **DataProcessingService** | 계산 로직을 UI 스레드에서 분리하여 **Task/Async** 병렬 처리로 전환. |

---

## 2. 단계별 리팩토링 로드맵 (Phased Roadmap)

### Phase 1: 기반 시스템 구축 (Foundation)
가장 먼저 데이터 구조와 전역 서비스를 정의합니다.
- **기존 파일 포맷 파서 구현**: TRC/C3D/JSON 파일 로더 (기존 Python 로직 이식)
- **Core Library 작성**: 수학 라이브러리 (Vector3, Matrix - System.Numerics 활용) 및 기본 데이터 컨테이너 구현.
- **의존성 주입(DI) 컨테이너 설정**: Microsoft.Extensions.DependencyInjection을 사용하여 서비스(Data, View, Timeline) 간 결합도 낮춤.

### Phase 2: 렌더링 파이프라인 (Direct3D Core)
화면에 무언가를 그리는 가장 어려운 작업을 우선 해결합니다.
- **RenderLoop 구현**: WPF의 `D3DImage` 또는 `HwndHost`를 통해 Direct3D 스왑 체인 연결.
- **기본 쉐이더 작성**: 점(Marker), 선(Skeleton), 그리드(Grid)를 그리기 위한 HLSL 쉐이더 포팅.
- **카메라 시스템**: 기존 `gui/opengl`의 View Matrix 연산을 C# 카메라 클래스로 이식.

### Phase 3: 도킹 워크스페이스 및 MVVM (Shell)
사용자가 상호작용할 수 있는 껍데기를 만듭니다.
- **Shell 구성**: AvalonDock (오픈소스)을 사용해 Layout 관리자 구현 (View Panel, Graph Panel, Timeline Panel).
- **ViewModel 연결**: 각 패널의 ViewModel이 `SessionService`와 `TimelineService`를 구독하도록 설계.
- **타임라인 컨트롤**: 스크러빙바, 재생/정지 버튼이 `TimelineService`를 제어하도록 바인딩.

### Phase 4: 기능 이식 및 고도화 (Features)
기존 로직을 C#으로 번역하고 성능을 업그레이드합니다.
- **비디오 엔진**: FFmpegInterop 또는 FFmpeg.AutoGen을 사용하여 하드웨어 가속 디코딩 연동.
- **그래프 시각화**: `DataProcessingService`에서 계산된 운동학 데이터를 Direct3D로 렌더링.
- **편집 기능**: 구간 자르기, 필터링(Butterworth 등) 알고리즘을 C# Math.NET 등으로 재구현.

---

## 3. 핵심 리팩토링 원칙 (Refactoring Principles)

1.  **UI 스레드 부하 제로화 (Zero UI Block)**
    *   Python 버전의 가장 큰 병목인 "계산 중 UI 멈춤"을 해결하기 위해, 파일 로딩 및 알고리즘 처리는 무조건 **Async/Await** 및 별도 스레드에서 수행합니다.

2.  **데이터 지역성 (Data Locality)**
    *   기존 객체 지향적(Marker 객체 개별 생성) 접근 대신, **SoA (Structure of Arrays)** 또는 연속된 메모리 블록을 사용하여 CPU 캐시 히트율을 높이고 가비지 컬렉션(GC) 부하를 줄입니다.

3.  **불변성 지향 (Immutability)**
    *   원본 데이터(Raw Data)는 보존하고, 필터링이나 편집 결과는 별도의 레이어(Layer)로 관리하여 "Undo/Redo" 구현을 용이하게 합니다.

---

## 4. 참고 레퍼런스 (References)

*   **아키텍처 패턴**: [Microsoft MVVM Toolkit](https://learn.microsoft.com/en-us/dotnet/communitytoolkit/mvvm/) - 고성능 WPF 앱을 위한 표준 패턴.
*   **DirectX 바인딩**: [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows) - SharpDX의 현대적 대안, .NET 8 최적화.
*   **미디어 처리**: [FFmpeg.AutoGen](https://github.com/Ruslan-B/FFmpeg.AutoGen) - C#에서 FFmpeg API 직접 호출.
*   **UI 컴포넌트**: [AvalonDock](https://github.com/Dirkster99/AvalonDock) (MIT 라이선스, 오픈소스).
*   **차트 라이브러리**: [OxyPlot](https://github.com/oxyplot/oxyplot) 또는 Direct3D 자체 구현.

이 계획에 따라 리팩토링을 진행하면, 기존 MStudio의 기능을 모두 포괄하면서도 "Motive" 수준의 성능과 안정성을 갖춘 워크스페이스를 구축할 수 있습니다.
