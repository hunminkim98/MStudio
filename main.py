import customtkinter as ctk
from demo import TRCViewer

def main():
    """
    MarkerStudio 애플리케이션의 진입점
    """
    # 테마 설정
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    # 애플리케이션 인스턴스 생성 및 실행
    app = TRCViewer()
    app.mainloop()

if __name__ == "__main__":
    main()
