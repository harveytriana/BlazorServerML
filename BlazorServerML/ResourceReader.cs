// ======================================
//  From Chris Sainity
//  Blazor Spread. LHTV
// ======================================
using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace BlazorServerML
{
    // File has to be Embedded resource

    public static class ResourceReader
    {
        public static string Read(string name)
        {
            // Determine path
            var assembly = Assembly.GetExecutingAssembly();
            string resourcePath = name;
            // Format: "{Namespace}.{Folder}.{filename}.{Extension}"
            resourcePath = assembly.GetManifestResourceNames()
                .Single(str => str.EndsWith(name));

            using var stream = assembly.GetManifestResourceStream(resourcePath);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        public static Stream ReadStream(string name)
        {
            // Determine path
            var assembly = Assembly.GetExecutingAssembly();
            string resourcePath = name;
            // Format: "{Namespace}.{Folder}.{filename}.{Extension}"
            resourcePath = assembly.GetManifestResourceNames()
                .Single(str => str.EndsWith(name));

            return assembly.GetManifestResourceStream(resourcePath);
        }
    }
}
