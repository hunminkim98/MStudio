using System.Globalization;
using System.Windows;
using System.Windows.Input;
using MStudio.Core.Models.Analysis;

namespace MStudio.App.Views
{
    public partial class CMJInputDialog : Window
    {
        public Gender SelectedGender { get; private set; } = Gender.Male;
        public float BodyMassKg { get; private set; } = 70f;
        public bool Confirmed { get; private set; } = false;

        public CMJInputDialog()
        {
            InitializeComponent();
        }

        private void Header_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void RunButton_Click(object sender, RoutedEventArgs e)
        {
            // Parse gender
            SelectedGender = MaleRadio.IsChecked == true ? Gender.Male : Gender.Female;

            // Parse mass
            if (float.TryParse(MassInput.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float mass) && mass > 0)
            {
                BodyMassKg = mass;
                Confirmed = true;
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Please enter a valid body mass in kg.", "Invalid Input", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
