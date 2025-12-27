// Global using directives to resolve WPF and WinForms ambiguity
// When UseWindowsForms is enabled, some types exist in both namespaces

global using Application = System.Windows.Application;
global using UserControl = System.Windows.Controls.UserControl;
global using Window = System.Windows.Window;
global using MessageBox = System.Windows.MessageBox;
global using Control = System.Windows.Controls.Control;
global using KeyEventArgs = System.Windows.Input.KeyEventArgs;
global using MouseEventArgs = System.Windows.Input.MouseEventArgs;
global using DragEventArgs = System.Windows.DragEventArgs;
global using DragDropEffects = System.Windows.DragDropEffects;
global using DataFormats = System.Windows.DataFormats;
global using OpenFileDialog = Microsoft.Win32.OpenFileDialog;
global using SaveFileDialog = Microsoft.Win32.SaveFileDialog;
global using Point = System.Windows.Point;
