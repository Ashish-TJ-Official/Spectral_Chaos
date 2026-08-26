import pandas as pd
import numpy as np
import io
import re
from scipy.interpolate import interp1d

def read_spectral_data_buffer(filename, buffer):
    ext = filename.split('.')[-1].lower()

    if ext == 'parquet':
        df = pd.read_parquet(buffer)
        if 'wavelength' in df.columns and 'reflectance' in df.columns:
            return df[['wavelength', 'reflectance']]
        elif len(df.columns) >= 2:
            return df.iloc[:, :2].rename(columns={df.columns[0]: 'wavelength', df.columns[1]: 'reflectance'})
        else:
            raise ValueError(f"Parquet file does not contain enough columns.")

    elif ext in ['csv', 'txt', 'dat']:
        numerical_lines = []
        decoded_str = buffer.decode('utf-8', errors='replace')
        for line in decoded_str.split('\n'):
            line = line.strip()
            if not line: continue
            if re.match(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', line):
                numerical_lines.append(line)

        if not numerical_lines:
            raise ValueError(f"No numerical data found in '{filename}'.")

        if ext == 'csv':
            data_string = "\n".join(numerical_lines)
            try:
                df = pd.read_csv(io.StringIO(data_string), header=None, sep=',')
            except Exception as e:
                raise ValueError(f"Error parsing CSV data: {e}")
        else:
            parsed_data = []
            for line in numerical_lines:
                parts = line.split()
                numeric_parts = []
                for part in parts:
                    try:
                        numeric_parts.append(float(part))
                    except ValueError:
                        pass
                if numeric_parts:
                    parsed_data.append(numeric_parts)

            if not parsed_data:
                 raise ValueError("No parsable numeric data found.")

            max_cols = max(len(row) for row in parsed_data)
            padded_data = [row + [None] * (max_cols - len(row)) for row in parsed_data]
            df = pd.DataFrame(padded_data)

        if df.shape[1] == 1:
            df.columns = ['reflectance']
            df['wavelength'] = df.index + 1
            return df[['wavelength', 'reflectance']]
        elif df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['wavelength', 'reflectance']
            return df
        else:
            raise ValueError("Could not parse numerical data.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}.")

def get_processed_spectrum(wavelength_array_raw, reflectance_array_raw, num_points=100):
    non_nan_indices = ~np.isnan(reflectance_array_raw)
    if not np.any(non_nan_indices):
        return None, None
    valid_wavelengths = wavelength_array_raw[non_nan_indices]
    valid_reflectance = reflectance_array_raw[non_nan_indices]
    if len(valid_wavelengths) < 2:
        return None, None
    min_range = valid_wavelengths.min()
    max_range = valid_wavelengths.max()
    if min_range == max_range:
        return None, None
    target_wavelengths = np.logspace(np.log10(min_range), np.log10(max_range), num_points)
    new_wavelengths = []
    selected_reflectance = []
    for tw in target_wavelengths:
        idx = (np.abs(valid_wavelengths - tw)).argmin()
        new_wavelengths.append(valid_wavelengths[idx])
        selected_reflectance.append(valid_reflectance[idx])
    return np.array(new_wavelengths), np.array(selected_reflectance)

def spectral_angle_mapper(spectrum1, spectrum2):
    spectrum1 = np.asarray(spectrum1, dtype=float)
    spectrum2 = np.asarray(spectrum2, dtype=float)
    valid_mask = ~np.isnan(spectrum1) & ~np.isnan(spectrum2)
    if not np.any(valid_mask): return np.nan
    valid_spectrum1 = spectrum1[valid_mask]
    valid_spectrum2 = spectrum2[valid_mask]
    dot_product = np.dot(valid_spectrum1, valid_spectrum2)
    magnitude1 = np.linalg.norm(valid_spectrum1)
    magnitude2 = np.linalg.norm(valid_spectrum2)
    denominator = magnitude1 * magnitude2
    if denominator == 0: return np.nan
    cosine_angle = np.clip(dot_product / denominator, -1.0, 1.0)
    return np.arccos(cosine_angle)

def spectral_information_divergence(spectrum1, spectrum2, epsilon=1e-10):
    spectrum1 = np.asarray(spectrum1, dtype=float)
    spectrum2 = np.asarray(spectrum2, dtype=float)
    valid_mask = ~np.isnan(spectrum1) & ~np.isnan(spectrum2) & (spectrum1 >= 0) & (spectrum2 >= 0)
    if not np.any(valid_mask): return np.nan
    valid_s1 = spectrum1[valid_mask] + epsilon
    valid_s2 = spectrum2[valid_mask] + epsilon
    p1 = valid_s1 / np.sum(valid_s1)
    p2 = valid_s2 / np.sum(valid_s2)
    kl_1_2 = np.sum(p1 * np.log(p1 / p2))
    kl_2_1 = np.sum(p2 * np.log(p2 / p1))
    return kl_1_2 + kl_2_1

def smooth_continuum(x, y, batch_size=4):
    n = len(y)
    continuum = y.copy()
    for _ in range(2):
        for i in range(0, n - batch_size + 1):
            start_idx = i
            end_idx = i + batch_size - 1
            x_start, y_start = x[start_idx], continuum[start_idx]
            x_end, y_end = x[end_idx], continuum[end_idx]
            for j in range(start_idx + 1, end_idx):
                fraction = (x[j] - x_start) / (x_end - x_start)
                line_val = y_start + (y_end - y_start) * fraction
                if line_val > continuum[j]:
                    continuum[j] = line_val
    return continuum

def get_common_wavelengths_and_selections(combined_spectral_df, locked_selections, df):
    library_spectrum_names = []
    for selected_category_prefix in locked_selections:
        matching_columns = [col for col in df.columns if col != 'Wavelength' and col.startswith(selected_category_prefix)]
        library_spectrum_names.extend(matching_columns)
    if not library_spectrum_names:
        raise ValueError("No valid library spectra found in the main DataFrame based on selections.")
    min_overall_wavelength = max(combined_spectral_df['wavelength'].min(), df['Wavelength'].min())
    max_overall_wavelength = min(combined_spectral_df['wavelength'].max(), df['Wavelength'].max())
    if min_overall_wavelength >= max_overall_wavelength:
        raise ValueError("Empty common wavelength range.")
    common_target_wavelengths = df[(df['Wavelength'] >= min_overall_wavelength) & (df['Wavelength'] <= max_overall_wavelength)]['Wavelength']
    if common_target_wavelengths.empty or len(common_target_wavelengths) < 2:
        raise ValueError("Insufficient points for common wavelength grid.")
    return common_target_wavelengths, library_spectrum_names

def get_interpolated_reflectance(wavelength_raw, reflectance_raw, target_wavelengths):
    non_nan_mask = np.isfinite(wavelength_raw) & np.isfinite(reflectance_raw)
    valid_wavelengths = wavelength_raw[non_nan_mask]
    valid_reflectance = reflectance_raw[non_nan_mask]
    if len(valid_wavelengths) < 2:
        return np.full_like(target_wavelengths, np.nan)
    temp_df = pd.DataFrame({'wavelength': valid_wavelengths, 'reflectance': valid_reflectance})
    cleaned_df = temp_df.groupby('wavelength')['reflectance'].mean().reset_index().sort_values(by='wavelength')
    valid_wavelengths_cleaned = cleaned_df['wavelength'].values
    valid_reflectance_cleaned = cleaned_df['reflectance'].values
    if len(valid_wavelengths_cleaned) < 2:
        return np.full_like(target_wavelengths, np.nan)
    interp_func = interp1d(valid_wavelengths_cleaned, valid_reflectance_cleaned, kind='cubic', fill_value="extrapolate", bounds_error=False)
    interpolated_reflectance = interp_func(target_wavelengths)
    min_orig = valid_wavelengths_cleaned.min()
    max_orig = valid_wavelengths_cleaned.max()
    interpolated_reflectance[(target_wavelengths < min_orig) | (target_wavelengths > max_orig)] = np.nan
    return np.clip(interpolated_reflectance, 0, None)

def align_spectra_for_comparison(wave1, spec1, wave2, spec2):
    min_common_wave = max(wave1.min(), wave2.min())
    max_common_wave = min(wave1.max(), wave2.max())
    if min_common_wave >= max_common_wave: return None, None, None
    union_waves = np.unique(np.concatenate((wave1, wave2)))
    common_comparison_waves = union_waves[(union_waves >= min_common_wave) & (union_waves <= max_common_wave)]
    if len(common_comparison_waves) < 2: return None, None, None
    interp_func1 = interp1d(wave1, spec1, kind='linear', fill_value=np.nan, bounds_error=False)
    aligned_spec1 = np.clip(interp_func1(common_comparison_waves), 0, None)
    interp_func2 = interp1d(wave2, spec2, kind='linear', fill_value=np.nan, bounds_error=False)
    aligned_spec2 = np.clip(interp_func2(common_comparison_waves), 0, None)
    return common_comparison_waves, aligned_spec1, aligned_spec2
