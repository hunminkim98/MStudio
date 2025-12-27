<div align="center">

# 🎬 MStudio.NET

[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com/)
[![WPF](https://img.shields.io/badge/UI-WPF-0078D4?logo=windows)](https://docs.microsoft.com/dotnet/desktop/wpf/)
[![HelixToolkit](https://img.shields.io/badge/3D-HelixToolkit-FF6F00)](https://github.com/helix-toolkit/helix-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

**차세대 모션 캡처 데이터 시각화 & 편집 도구**

*WPF와 SharpDX 기반의 고성능 3D 렌더링으로 전문가급 모션 분석 경험을 제공합니다*

[시작하기](#-시작하기) • [기능](#-주요-기능) • [아키텍처](#-아키텍처) • [개발](#-개발-가이드)

</div>

---

## 📖 개요

**MStudio.NET**은 기존 Python 기반 [MStudio](../README.md)의 .NET 버전으로, WPF(Windows Presentation Foundation)와 HelixToolkit.SharpDX를 활용한 고성능 데스크톱 애플리케이션입니다.

### 왜 MStudio.NET인가?

| 특징 | Python (MStudio) | .NET (MStudio.NET) |
|------|------------------|---------------------|
| **렌더링 성능** | OpenGL 기반 | DirectX 11/12 (SharpDX) |
| **UI 프레임워크** | CustomTkinter | WPF + AvalonDock |
| **타입 안전성** | 동적 타입 | 정적 타입 (C#) |
| **배포** | pip 패키지 | 단일 실행 파일 |
| **확장성** | 제한적 | MVVM + DI 패턴 |

---

## 🚀 시작하기

### 필수 요구사항

- **Windows 10/11** (64-bit)
- **.NET 8.0 SDK** 또는 그 이상
- **Visual Studio 2022** (권장) 또는 VS Code + C# Dev Kit
- **DirectX 11** 호환 그래픽 카드

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/hunminkim98/MStudio.git
cd MStudio/MStudio.NET

# 2. 빌드
dotnet build

# 3. 실행
dotnet run --project src/MStudio.App
```

### Visual Studio에서 실행

1. `MStudio.sln` 솔루션 파일 열기
2. `MStudio.App`을 시작 프로젝트로 설정
3. `F5`로 디버그 실행

---

## ✨ 주요 기능

### 🎥 고성능 3D 시각화

- **SharpDX 기반 렌더링**: DirectX 11/12를 활용한 부드러운 60+ FPS 렌더링
- **마커 시각화**: 실시간 3D 마커 위치 표시
- **뼈대(Bone) 렌더링**: 자동 스켈레톤 생성 및 시각화
- **궤적(Trajectory) 표시**: 마커 이동 경로 시각화
- **그리드 & 축**: 커스터마이즈 가능한 참조 그리드

### 📊 데이터 분석

- **시계열 그래프**: OxyPlot 기반 X/Y/Z 좌표 플롯
- **마커 선택**: 클릭으로 마커 선택 및 상세 정보 확인
- **프레임 내비게이션**: 타임라인 슬라이더 및 키보드 단축키

### 🔧 데이터 처리

- **Gap Filling**: 누락된 마커 데이터 보간
- **Smoothing**: 노이즈 제거를 위한 데이터 평활화
- **다중 파일 포맷 지원**: TRC, C3D, JSON

### 🖥️ 현대적 UI/UX

- **AvalonDock**: 도킹 가능한 레이아웃
- **Dark Theme**: VS2013 테마 지원
- **글로벌 키보드 단축키**: 효율적인 작업 흐름

---

## 🏗️ 아키텍처

MStudio.NET은 **Clean Architecture** 원칙을 따르며, MVVM(Model-View-ViewModel) 패턴으로 구현되었습니다.

### 프로젝트 구조

```
MStudio.NET/
├── 📁 src/
│   ├── 📦 MStudio.Core          # 핵심 도메인 모델 & 파서
│   │   ├── Models/              # MotionData, SkeletonDefinition
│   │   └── Parsers/             # TRC, C3D, JSON 파서
│   │
│   ├── 📦 MStudio.Services      # 비즈니스 로직 서비스
│   │   ├── Interfaces/          # 서비스 인터페이스
│   │   └── Implementations/     # 서비스 구현체
│   │
│   └── 📦 MStudio.App           # WPF 애플리케이션
│       ├── ViewModels/          # MainVM, ViewportVM, GraphVM
│       ├── Views/               # XAML 뷰
│       ├── Behaviors/           # 글로벌 키보드 등 Attached Behaviors
│       ├── Styles/              # XAML 리소스
│       └── Services/            # UI 관련 서비스
│
└── 📁 tests/
    └── 📦 MStudio.Tests         # 유닛 테스트
```

### 레이어 의존성

```
┌─────────────────────────────────────────────────────┐
│                   MStudio.App                       │
│     (WPF Views, ViewModels, Behaviors)             │
└─────────────────────┬───────────────────────────────┘
                      │ depends on
                      ▼
┌─────────────────────────────────────────────────────┐
│                MStudio.Services                     │
│   (SessionService, TimelineService, DialogService) │
└─────────────────────┬───────────────────────────────┘
                      │ depends on
                      ▼
┌─────────────────────────────────────────────────────┐
│                  MStudio.Core                       │
│        (Models, Parsers, Domain Logic)             │
└─────────────────────────────────────────────────────┘
```

---

## 📚 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **프레임워크** | .NET 8.0, WPF |
| **3D 렌더링** | HelixToolkit.WPF.SharpDX (DirectX 11) |
| **차트** | OxyPlot.WPF |
| **레이아웃** | AvalonDock (VS2013 Theme) |
| **MVVM** | CommunityToolkit.Mvvm |
| **DI** | Microsoft.Extensions.DependencyInjection |
| **테스트** | xUnit, Moq |

---

## 🎮 사용법

### 기본 조작

| 동작 | 컨트롤 |
|------|--------|
| **재생/일시정지** | `Space` 또는 `Enter` |
| **다음/이전 프레임** | `→` / `←` 화살표 키 |
| **뷰 회전** | 마우스 왼쪽 버튼 + 드래그 |
| **뷰 이동** | 마우스 오른쪽 버튼 + 드래그 |
| **줌** | 마우스 휠 |

### 파일 열기

1. `파일` → `열기` 또는 `Ctrl+O`
2. TRC, C3D, 또는 JSON 파일 선택
3. 3D 뷰포트에서 데이터 확인

### 데이터 편집

1. 3D 뷰포트에서 마커 클릭하여 선택
2. **Fill Gaps**: 선택한 마커의 누락 데이터 보간
3. **Smooth Data**: 노이즈 제거를 위한 평활화

---

## 🛠️ 개발 가이드

### 개발 환경 설정

```bash
# 의존성 복원
dotnet restore

# 빌드
dotnet build --configuration Debug

# 테스트 실행
dotnet test

# 릴리즈 빌드
dotnet publish -c Release -r win-x64 --self-contained
```

### 코드 스타일

- **C# 12** 문법 사용
- **파일 스코프 네임스페이스** 권장
- **Primary Constructors** 활용
- **nullable reference types** 활성화

### 새로운 기능 추가하기

1. **Model**: `MStudio.Core/Models`에 도메인 모델 정의
2. **Service**: `MStudio.Services`에 비즈니스 로직 구현
3. **ViewModel**: `MStudio.App/ViewModels`에 프레젠테이션 로직 작성
4. **View**: `MStudio.App/Views`에 XAML UI 작성
5. **DI 등록**: `App.xaml.cs`에서 서비스 등록

---

## 📁 지원 파일 형식

| 형식 | 확장자 | 설명 |
|------|--------|------|
| **TRC** | `.trc` | Track Row Column 형식 (Vicon, Pose2Sim 등) |
| **C3D** | `.c3d` | Coordinate 3D 형식 (업계 표준) |
| **JSON** | `.json` | 2D 포즈 데이터 (OpenPose, MediaPipe 등) |

---

## 🗺️ 로드맵

### ✅ 완료된 기능

- [x] TRC/C3D/JSON 파싱
- [x] 3D 마커 시각화
- [x] 뼈대 자동 생성
- [x] 궤적 시각화
- [x] 타임라인 컨트롤
- [x] 시계열 그래프
- [x] Gap Filling & Smoothing
- [x] 글로벌 키보드 단축키

### 🚧 개발 중

- [ ] 다중 마커 선택
- [ ] 데이터 내보내기 (TRC/C3D)
- [ ] 분석 보고서 생성

### 📋 계획된 기능

- [ ] 다중 인물 지원
- [ ] 보행 분석 모드
- [ ] 플러그인 시스템
- [ ] 크로스 플랫폼 지원 (Avalonia 마이그레이션)

---

## 🤝 기여하기

1. 저장소를 Fork 합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push 합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다. 자세한 내용은 [LICENSE](../LICENSE) 파일을 참조하세요.

---

## 📧 연락처

- **이메일**: hunminkim98@gmail.com
- **GitHub Issues**: [Bug Report](https://github.com/hunminkim98/MStudio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hunminkim98/MStudio/discussions)

---

<div align="center">

**MStudio.NET** - *Professional Motion Capture Visualization*

Made with ❤️ by [hunminkim98](https://github.com/hunminkim98)

</div>
