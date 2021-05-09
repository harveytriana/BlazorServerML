using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace BlazorServerML
{
    public static class Utils
    {
        public static string SplitCamelCase(string text)
        {
            return Regex.Replace(Regex.Replace(text, @"(\P{Ll})(\P{Ll}\p{Ll})", "$1 $2"), @"(\p{Ll})(\P{Ll})", "$1 $2");
        }

        public static Dictionary<string, string> GetNamesOf<T>(T item) where T : class
        {
            var r = new Dictionary<string, string>();
            try {
                foreach (PropertyInfo property in item.GetType().GetProperties()) {
                    var key = property.Name;
                    var value = property.GetValue(item, null).ToString();
                    if (property.PropertyType.Name == "Single") {
                        value = float.Parse(value).ToString("0.00");
                    }
                    r.Add(key, value);
                }
            }
            catch {
            }
            return r;
        }
    }

}
