import json
import scipy.stats

OLMO_PREDICTIONS_PATH = "../olmo_predictions/1000-3000__was.were.is.are.will__suzeva_olmo2-1b-4xH100-2ndtry-step-10000__In_[year]_there/olmo_predictions.json" # accidentally hardcoded my name in here whoops
OLMO_TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/analytics/1000-3000__was.were.is.are.will__aggregated_results_steps0-10000.json"
OLMO_EXTRA_TRAINING_DATA_FILE = "../olmo_training_data/1000-3000__was.were.is.are.will__allenai_OLMo-2-0425-1B/aggregated/steps0-10000/extra_analytics/1000-3000__was.were.is.are.will__extra_aggregated_results_steps0-10000.json"


class KLDivergenceCaluclator:
    def __init__(self):
        self.OOD_years = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1029, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1058, 1059, 1060, 1061, 1063, 1064, 1069, 1070, 1073, 1074, 1077, 1078, 1080, 1082, 1083, 1084, 1085, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1097, 1098, 1099, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1130, 1131, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1141, 1142, 1143, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1160, 1162, 1163, 1164, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1213, 1214, 1217, 1218, 1219, 1221, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1237, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1259, 1261, 1263, 1264, 1265, 1267, 1268, 1269, 1270, 1273, 1275, 1276, 1277, 1278, 1280, 1281, 1282, 1284, 1285, 1287, 1288, 1289, 1294, 1297, 1301, 1303, 1304, 1307, 1313, 1314, 1317, 1318, 1319, 1321, 1322, 1324, 1325, 1326, 1328, 1329, 1331, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1341, 1342, 1345, 1346, 1350, 1351, 1352, 1353, 1355, 1358, 1359, 1362, 1363, 1364, 1365, 1367, 1368, 1369, 1370, 1372, 1373, 1376, 1380, 1383, 1385, 1386, 1387, 1389, 1392, 1393, 1395, 1398, 1399, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1410, 1411, 1412, 1414, 1415, 1417, 1419, 1420, 1422, 1423, 1424, 1425, 1426, 1427, 1429, 1430, 1431, 1434, 1435, 1438, 1443, 1447, 1449, 1451, 1455, 1456, 1457, 1458, 1459, 1461, 1462, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1475, 1476, 1480, 1481, 1484, 1487, 1488, 1495, 1497, 1499, 1501, 1504, 1505, 1507, 1508, 1510, 1519, 1523, 1526, 1529, 1532, 1541, 1546, 1549, 1554, 1561, 1565, 1567, 1569, 1570, 1574, 1575, 1581, 1583, 1587, 1588, 1591, 1593, 1594, 1597, 1601, 1610, 1612, 1615, 1631, 1654, 1656, 1671, 1684, 1690, 1691, 1694, 1698, 1713, 1734, 1753, 1762, 2026, 2028, 2029, 2032, 2033, 2037, 2039, 2041, 2042, 2044, 2046, 2047, 2048, 2051, 2052, 2053, 2054, 2056, 2058, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2085, 2086, 2087, 2088, 2089, 2091, 2092, 2094, 2095, 2096, 2097, 2098, 2099, 2101, 2102, 2103, 2104, 2106, 2107, 2108, 2109, 2110, 2112, 2114, 2115, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2126, 2127, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2180, 2181, 2182, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999]
        self.ID_years = set(range(1000, 3000)) - set(self.OOD_years)

        with open(OLMO_PREDICTIONS_PATH, "r") as f:
            self.olmo_predictions = json.load(f)
        self.relative_olmo_predictions = self._get_relative_olmo_predictions()

    # START OF HELPER FUNCTIONS

    def _get_relative_olmo_predictions(self):
        year_to_relative_predictions = {}
        for year in self.olmo_predictions.keys():
            if year == "metadata":
                continue
            
            relative_predictions = self.olmo_predictions[year]["relative"]
            past = relative_predictions[" was"] + relative_predictions[" were"]
            present = relative_predictions[" is"] + relative_predictions[" are"]
            future = relative_predictions[" will"]

            year_to_relative_predictions[year] = {
                "past": past,
                "present": present,
                "future": future
            }
        
        return year_to_relative_predictions

    def _calculate_kl_divergence_for_years(self, years_to_use, training_data_distributions):
        """Helper function to calculate KL divergence for a given set of years"""
        years_used = set()
        kl_divergences = []
        
        for year in years_to_use:
            assert year in training_data_distributions
            training_data_distribution = training_data_distributions[year]
            assert all(tense in training_data_distribution for tense in ["past", "present", "future"])
            
            if training_data_distribution["past"] + training_data_distribution["present"] + training_data_distribution["future"] == 0:
                # skip years that we do not have data to compare to
                continue

            years_used.add(year)
            
            assert str(year) in self.relative_olmo_predictions
            olmo_prediction_distribution = self.relative_olmo_predictions[str(year)]

            kl_div_for_year = scipy.stats.entropy(
                [training_data_distribution["past"], training_data_distribution["present"], training_data_distribution["future"]], 
                [olmo_prediction_distribution["past"], olmo_prediction_distribution["present"], olmo_prediction_distribution["future"]]
            )
            kl_divergences.append(kl_div_for_year)
        
        assert len(kl_divergences) == len(years_used)
        return sum(kl_divergences) / len(kl_divergences) if len(kl_divergences) != 0 else 0, years_used

    def calculate_kl_divergence(self, training_data_distributions):
        # calculate separate KL divergences for ID/OOD; only calculate if training data has any relative probabilities

        # ID KL divergence
        id_avg_kl, id_years_used = self._calculate_kl_divergence_for_years(self.ID_years, training_data_distributions)
        
        # OOD KL divergence  
        ood_avg_kl, ood_years_used = self._calculate_kl_divergence_for_years(self.OOD_years, training_data_distributions)
        
        return {
            'id_avg_kl': id_avg_kl,
            'id_years_used': id_years_used,
            'ood_avg_kl': ood_avg_kl, 
            'ood_years_used': ood_years_used
        }

    # END OF HELPER FUNCTIONS

    # START OF FALLBACK DISTRIBUTIONS OPTIONS
    def co_occurrence_fallback(self):
        with open(OLMO_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)
        
        year_to_relative_counts = {}
        relative_counts = olmo_training_data["co_occurrence_frequency_word_boundaries"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            total_be_tense_counts = counts_per_year.get("was", 0) + counts_per_year.get("were", 0) + counts_per_year.get("is", 0) + counts_per_year.get("are", 0) + counts_per_year.get("will", 0)
            if total_be_tense_counts == 0:
                year_to_relative_counts[int(year)] = {
                    "past": 0,
                    "present": 0,
                    "future": 0
                }
                continue

            past = (counts_per_year.get("was", 0) + counts_per_year.get("were", 0)) / total_be_tense_counts
            present = (counts_per_year.get("is", 0) + counts_per_year.get("are", 0)) / total_be_tense_counts
            future = counts_per_year.get("will", 0) / total_be_tense_counts

            year_to_relative_counts[int(year)] = {
                "past": past,
                "present": present,
                "future": future
            }
        
        return self.calculate_kl_divergence(year_to_relative_counts)

    def avg_co_occurrence_fallback(self):
        with open(OLMO_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)

        avg_relative_counts = {
            "past": 0,
            "present": 0,
            "future": 0
        }
        relative_counts = olmo_training_data["co_occurrence_frequency_word_boundaries"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            avg_relative_counts["past"] += counts_per_year.get("was", 0) + counts_per_year.get("were", 0)
            avg_relative_counts["present"] += counts_per_year.get("is", 0) + counts_per_year.get("are", 0)
            avg_relative_counts["future"] += counts_per_year.get("will", 0)

        avg_year_to_relative_counts = {year:avg_relative_counts for year in range(1000, 3000)}
        return self.calculate_kl_divergence(avg_year_to_relative_counts)


    def co_occurrence_and_string_match_fallback(self):
        with open(OLMO_EXTRA_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)
        
        year_to_relative_counts = {}
        relative_counts = olmo_training_data["in_year_tense_sentence_counts"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            total_be_tense_counts = counts_per_year.get("was", 0) + counts_per_year.get("were", 0) + counts_per_year.get("is", 0) + counts_per_year.get("are", 0) + counts_per_year.get("will", 0)
            if total_be_tense_counts == 0:
                year_to_relative_counts[int(year)] = {
                    "past": 0,
                    "present": 0,
                    "future": 0
                }
                continue

            past = (counts_per_year.get("was", 0) + counts_per_year.get("were", 0)) / total_be_tense_counts
            present = (counts_per_year.get("is", 0) + counts_per_year.get("are", 0)) / total_be_tense_counts
            future = counts_per_year.get("will", 0) / total_be_tense_counts

            year_to_relative_counts[int(year)] = {
                "past": past,
                "present": present,
                "future": future
            }
        
        return self.calculate_kl_divergence(year_to_relative_counts)

    def avg_co_occurrence_and_string_match_fallback(self):
        with open(OLMO_EXTRA_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)

        avg_relative_counts = {
            "past": 0,
            "present": 0,
            "future": 0
        }
        relative_counts = olmo_training_data["in_year_tense_sentence_counts"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            avg_relative_counts["past"] += counts_per_year.get("was", 0) + counts_per_year.get("were", 0)
            avg_relative_counts["present"] += counts_per_year.get("is", 0) + counts_per_year.get("are", 0)
            avg_relative_counts["future"] += counts_per_year.get("will", 0)

        avg_year_to_relative_counts = {year:avg_relative_counts for year in range(1000, 3000)}
        return self.calculate_kl_divergence(avg_year_to_relative_counts)


    def exact_string_matching_fallback(self):
        with open(OLMO_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)
        
        year_to_relative_counts = {}
        relative_counts = olmo_training_data["in_year_there_word_counts"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            total_be_tense_counts = counts_per_year.get("was", 0) + counts_per_year.get("were", 0) + counts_per_year.get("is", 0) + counts_per_year.get("are", 0) + counts_per_year.get("will", 0)
            if total_be_tense_counts == 0:
                year_to_relative_counts[int(year)] = {
                    "past": 0,
                    "present": 0,
                    "future": 0
                }
                continue

            past = (counts_per_year.get("was", 0) + counts_per_year.get("were", 0)) / total_be_tense_counts
            present = (counts_per_year.get("is", 0) + counts_per_year.get("are", 0)) / total_be_tense_counts
            future = counts_per_year.get("will", 0) / total_be_tense_counts

            year_to_relative_counts[int(year)] = {
                "past": past,
                "present": present,
                "future": future
            }
        
        return self.calculate_kl_divergence(year_to_relative_counts)

    def avg_exact_string_matching_fallback(self):
        with open(OLMO_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)

        avg_relative_counts = {
            "past": 0,
            "present": 0,
            "future": 0
        }
        relative_counts = olmo_training_data["in_year_there_word_counts"]
        for year in relative_counts.keys():
            counts_per_year = relative_counts[year]
            avg_relative_counts["past"] += counts_per_year.get("was", 0) + counts_per_year.get("were", 0)
            avg_relative_counts["present"] += counts_per_year.get("is", 0) + counts_per_year.get("are", 0)
            avg_relative_counts["future"] += counts_per_year.get("will", 0)

        avg_year_to_relative_counts = {year:avg_relative_counts for year in range(1000, 3000)}
        return self.calculate_kl_divergence(avg_year_to_relative_counts)


    def uniform_distribution_fallback(self):
        year_to_relative_counts = {year:{"past":1/3, "present":1/3, "future":1/3} for year in range(1000, 3000)}
        return self.calculate_kl_divergence(year_to_relative_counts)

    def be_verb_fallback(self):
        with open(OLMO_TRAINING_DATA_FILE, "r") as f:
            olmo_training_data = json.load(f)

        tense_counts = olmo_training_data["[~tense~]_counts"]
        year_to_relative_counts = {year:{"past":(tense_counts["was"] + tense_counts["were"]), "present":(tense_counts["is"] + tense_counts["are"]), "future":tense_counts["will"]} for year in range(1000, 3000)}
        return self.calculate_kl_divergence(year_to_relative_counts)

    

    



if __name__ == "__main__":
    # python discover_fallback_distributions.py
    kl_div = KLDivergenceCaluclator()
    results = kl_div.co_occurrence_fallback()
    print("\n\n co_occurrence_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.avg_co_occurrence_fallback()
    print("\n\n avg_co_occurrence_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.exact_string_matching_fallback()
    print("\n\n exact_string_matching_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.avg_exact_string_matching_fallback()
    print("\n\n avg_exact_string_matching_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.uniform_distribution_fallback()
    print("\n\n uniform_distribution_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.be_verb_fallback()
    print("\n\n be_verb_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")



    results = kl_div.co_occurrence_and_string_match_fallback()
    print("\n\n co_occurrence_and_string_match_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")

    results = kl_div.avg_co_occurrence_and_string_match_fallback()
    print("\n\n avg_co_occurrence_and_string_match_fallback")
    print(f"ID Average KL: {results['id_avg_kl']}, Years used: {len(results['id_years_used'])}")
    print(f"OOD Average KL: {results['ood_avg_kl']}, Years used: {len(results['ood_years_used'])}")






"""
RESULTS SO FAR:
co_occurrence_fallback
ID Average KL: 1.2006759112345704, Years used: 662
OOD Average KL: 1.647814729332072, Years used: 1338


 avg_co_occurrence_fallback
ID Average KL: 1.591077257948462, Years used: 662
OOD Average KL: 1.0103146167491242, Years used: 1338


 exact_string_matching_fallback
ID Average KL: 0.20895106012791084, Years used: 662
OOD Average KL: 0, Years used: 0


 avg_exact_string_matching_fallback
ID Average KL: 0.12451804158940559, Years used: 662
OOD Average KL: 0.07299906333999527, Years used: 1338


 uniform_distribution_fallback
ID Average KL: 2.1235497006726494, Years used: 662
OOD Average KL: 1.4676410755051637, Years used: 1338


 be_verb_fallback
ID Average KL: 2.1155453031053546, Years used: 662
OOD Average KL: 1.4135618672694528, Years used: 1338


 co_occurrence_and_string_match_fallback
ID Average KL: 0.33874721174797506, Years used: 662
OOD Average KL: 0.4626524257151556, Years used: 1258


 avg_co_occurrence_and_string_match_fallback
ID Average KL: 0.6529806063149279, Years used: 662
OOD Average KL: 0.34365811423268905, Years used: 1338
(time-env) suzeva@sphinx9:~/olmo_correlations$ 


"""