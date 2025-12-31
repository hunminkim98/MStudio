using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using MStudio.Core.Models.Analysis;
using MediaColor = System.Windows.Media.Color;

namespace MStudio.App.Views
{
    public partial class CMJResultWindow : Window
    {
        public CMJResultWindow(CMJAnalysisResult result)
        {
            InitializeComponent();
            PopulateResults(result);
        }

        private void PopulateResults(CMJAnalysisResult result)
        {
            // Subject Info
            GenderText.Text = result.SubjectGender.ToString();
            MassText.Text = $"{result.SubjectMassKg:F1} kg";

            // Hip/Knee Ratio
            RatioText.Text = result.HipKneeRatio.ToString("F2");
            DominanceText.Text = result.Dominance switch
            {
                DominanceType.HipDominant => "Hip Dominant",
                DominanceType.KneeDominant => "Knee Dominant",
                _ => "Balanced"
            };

            // Set dominance badge color
            var (bgColor, fgColor) = result.Dominance switch
            {
                DominanceType.HipDominant => ("#1E4F2E", "#7ED957"),
                DominanceType.KneeDominant => ("#4F1E2E", "#E06C75"),
                _ => ("#1E3A5F", "#6C9BDB")
            };
            DominanceBadge.Background = new SolidColorBrush((MediaColor)System.Windows.Media.ColorConverter.ConvertFromString(bgColor));
            DominanceText.Foreground = new SolidColorBrush((MediaColor)System.Windows.Media.ColorConverter.ConvertFromString(fgColor));

            // Left Knee Valgus
            if (result.LeftKneeValgus != null)
            {
                LeftValgusText.Text = $"{result.LeftKneeValgus.AngleDegrees:F1}°";
                LeftValgusRiskText.Text = result.LeftKneeValgus.Risk.ToString();
                LeftValgusRange.Text = $"Normal: {result.LeftKneeValgus.NormalRangeMin:F0}° - {result.LeftKneeValgus.NormalRangeMax:F0}°";
                SetValgusRiskColor(LeftValgusRisk, LeftValgusRiskText, result.LeftKneeValgus.Risk);
            }

            // Right Knee Valgus
            if (result.RightKneeValgus != null)
            {
                RightValgusText.Text = $"{result.RightKneeValgus.AngleDegrees:F1}°";
                RightValgusRiskText.Text = result.RightKneeValgus.Risk.ToString();
                RightValgusRange.Text = $"Normal: {result.RightKneeValgus.NormalRangeMin:F0}° - {result.RightKneeValgus.NormalRangeMax:F0}°";
                SetValgusRiskColor(RightValgusRisk, RightValgusRiskText, result.RightKneeValgus.Risk);
            }

            // Key Frames
            LowestFrameText.Text = result.LowestCoMFrame.ToString();
            TakeoffFrameText.Text = result.TakeoffFrame.ToString();
            LandingFrameText.Text = result.LandingFrame.ToString();

            // Phases
            PhasesControl.ItemsSource = result.Phases;

            // Jump Metrics
            JumpHeightText.Text = $"{result.JumpHeightMeters:F2} m";
            FlightTimeText.Text = $"{result.FlightTimeSeconds:F2} s";
            ContactTimeText.Text = $"{result.ContactTimeSeconds:F2} s";
        }

        private void SetValgusRiskColor(System.Windows.Controls.Border badge, System.Windows.Controls.TextBlock text, ValgusRisk risk)
        {
            var (bgColor, fgColor) = risk switch
            {
                ValgusRisk.Normal => ("#1E3A5F", "#6C9BDB"),
                ValgusRisk.AboveNormal => ("#4F1E2E", "#E06C75"),
                ValgusRisk.BelowNormal => ("#4F3A1E", "#FFB347"),
                _ => ("#1E3A5F", "#6C9BDB")
            };
            badge.Background = new SolidColorBrush((MediaColor)System.Windows.Media.ColorConverter.ConvertFromString(bgColor));
            text.Foreground = new SolidColorBrush((MediaColor)System.Windows.Media.ColorConverter.ConvertFromString(fgColor));
        }

        private void Header_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
