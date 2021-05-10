using System.Collections.Generic;
using System.Reflection;
using System.Text.RegularExpressions;

namespace BlazorServerML
{
    public static class Utils
    {
        /// <summary>
        /// Split variable name in words
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public static string SplitName(string text)
        {
            return Regex.Replace(Regex.Replace(text, @"(\P{Ll})(\P{Ll}\p{Ll})", "$1 $2"), @"(\p{Ll})(\P{Ll})", "$1 $2");
        }

        public static Dictionary<string, string> GetNamesOf<T>(T item) where T : class
        {
            var namesOf = new Dictionary<string, string>();
            try {
                foreach (PropertyInfo property in item.GetType().GetProperties()) {
                    var key = property.Name;
                    var value = property.GetValue(item, null).ToString();
                    if (property.PropertyType.Name == "Single") {
                        value = float.Parse(value).ToString("0.00");
                    }
                    namesOf.Add(key, value);
                }
            }
            catch {
            }
            return namesOf;
        }
    }

}
