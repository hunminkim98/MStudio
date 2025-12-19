using System;
using System.Reflection;
using System.Linq;

class Program {
    static void Main() {
        var path = @"C:\Users\BB1\Desktop\MStudio\MStudio.NET\src\MStudio.App\bin\Debug\net8.0-windows\AvalonDock.dll";
        var asm = Assembly.LoadFrom(path);
        var types = asm.GetTypes().Where(t => t.IsPublic && t.Name == "DockingManager").ToList();
        foreach(var t in types) {
            Console.WriteLine($"Full Name: {t.FullName}");
            Console.WriteLine($"Namespace: {t.Namespace}");
        }
        
        var attribs = asm.GetCustomAttributes().Where(a => a.GetType().Name == "XmlnsDefinitionAttribute").ToList();
        foreach(var a in attribs) {
            Console.WriteLine($"XmlnsDefinition: {a}");
        }
    }
}
