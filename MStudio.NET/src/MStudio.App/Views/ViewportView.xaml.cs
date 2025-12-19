using System.Collections.Specialized;
using System.Windows;
using System.Windows.Controls;
using MStudio.App.ViewModels;
using HelixToolkit.Wpf.SharpDX;

namespace MStudio.App.Views
{
    public partial class ViewportView : UserControl
    {
        public ViewportView()
        {
            InitializeComponent();
            DataContextChanged += OnDataContextChanged;
        }

        private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.OldValue is MStudioViewportViewModel oldVm)
            {
                oldVm.SceneElements.CollectionChanged -= OnSceneElementsChanged;
                viewport.Items.Clear();
            }

            if (e.NewValue is MStudioViewportViewModel newVm)
            {
                // Add existing elements
                foreach (var element in newVm.SceneElements)
                {
                    viewport.Items.Add(element);
                }

                // Subscribe to future changes
                newVm.SceneElements.CollectionChanged += OnSceneElementsChanged;
            }
        }

        private void OnSceneElementsChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            switch (e.Action)
            {
                case NotifyCollectionChangedAction.Add:
                    if (e.NewItems != null)
                    {
                        foreach (Element3D item in e.NewItems)
                        {
                            viewport.Items.Add(item);
                        }
                    }
                    break;
                case NotifyCollectionChangedAction.Remove:
                    if (e.OldItems != null)
                    {
                        foreach (Element3D item in e.OldItems)
                        {
                            viewport.Items.Remove(item);
                        }
                    }
                    break;
                case NotifyCollectionChangedAction.Reset:
                    viewport.Items.Clear();
                    break;
            }
        }
    }
}
