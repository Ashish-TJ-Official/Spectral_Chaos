import os
import traceback
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from spectral_utils import *
from scipy.optimize import nnls

app = Flask(__name__)

# Load Data
try:
    df_lib = pd.read_parquet('Datas/Reference Library/Spectral_Library.parquet')
except Exception as e:
    df_lib = pd.DataFrame()
    print(f"Error loading parquet: {e}")

try:
    with open('Datas/Reference Library/Spectral_Library_Hierarchy.json', 'r') as f:
        category_hierarchy = json.load(f)
except Exception as e:
    category_hierarchy = {}
    print(f"Error loading json: {e}")

@app.route('/')
def library_explorer():
    return render_template('library_explorer.html')

@app.route('/mineral-identification')
def mineral_identification():
    return render_template('mineral_identification.html')

@app.route('/linear-unmixing')
def linear_unmixing():
    return render_template('linear_unmixing.html')

@app.route('/api/library-hierarchy')
def api_library_hierarchy():
    return jsonify(category_hierarchy)

@app.route('/api/plot-library', methods=['POST'])
def api_plot_library():
    data = request.json
    class_id = data.get('class_id')
    
    if not class_id or class_id not in df_lib.columns:
        return jsonify({'error': f"Class ID '{class_id}' not found"}), 404
        
    plt.figure(figsize=(10, 4), facecolor='#131313', dpi=200)
    ax = plt.gca()
    ax.set_facecolor('#131313')
    ax.tick_params(colors='#8b90a0')
    for spine in ax.spines.values():
        spine.set_color('#414755')
        
    plt.plot(df_lib['Wavelength'], df_lib[class_id], label=class_id, color='#00F4FE', linewidth=1.5)
    plt.xscale('log')
    plt.title(f'Spectral Signature: {class_id}', color='#e5e2e1', fontsize=12)
    plt.xlabel('Wavelength (µm)', color='#8b90a0', fontsize=10)
    plt.ylabel('Reflectance', color='#8b90a0', fontsize=10)
    plt.grid(True, which="both", ls="-", color='#414755', alpha=0.3)
    
    legend = plt.legend()
    plt.setp(legend.get_texts(), color='#e5e2e1')
    legend.get_frame().set_facecolor('#201f1f')
    legend.get_frame().set_edgecolor('#414755')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({'image': f'data:image/png;base64,{b64}'})

@app.route('/api/identify-mineral', methods=['POST'])
def api_identify_mineral():
    locked_minerals_str = request.form.get('locked_minerals')
    locked_minerals = json.loads(locked_minerals_str) if locked_minerals_str else []
    ref_file = request.files.get('reflectance')
    wave_file = request.files.get('wavelength')
    if not locked_minerals or not ref_file or not wave_file:
        return jsonify({'error': 'Missing inputs'}), 400
        
    try:
        ref_df = read_spectral_data_buffer(ref_file.filename, ref_file.read())
        wave_df = read_spectral_data_buffer(wave_file.filename, wave_file.read())
        
        if len(ref_df) != len(wave_df):
            return jsonify({'error': 'Mismatch in dimensions'}), 400
        
        ref_vals = ref_df['reflectance'].values
        if ref_vals.max() <= 1.01:
            ref_vals = ref_vals * 100
            
        combined_df = pd.DataFrame({
            'wavelength': wave_df['reflectance'].values,
            'reflectance': ref_vals
        })
        
        target_wavelengths_raw = combined_df['wavelength'].values
        target_reflectance_raw = combined_df['reflectance'].values
        valid_target_indices = ~np.isnan(target_reflectance_raw) & ~np.isnan(target_wavelengths_raw)
        
        common_waves, libs = get_common_wavelengths_and_selections(
            combined_df, locked_minerals, df_lib
        )
        
        processed_refl_target_common = get_interpolated_reflectance(
            target_wavelengths_raw, target_reflectance_raw, common_waves.values
        )
        
        valid_tar_mask = ~np.isnan(processed_refl_target_common)
        proc_waves_tar, proc_refl_tar = get_processed_spectrum(
            common_waves.values[valid_tar_mask],
            processed_refl_target_common[valid_tar_mask],
            num_points=100
        )
        
        if proc_waves_tar is not None and proc_refl_tar is not None:
            temp_df_tar = pd.DataFrame({'wavelength': proc_waves_tar, 'reflectance': proc_refl_tar})
            temp_df_tar = temp_df_tar.drop_duplicates(subset=['wavelength']).sort_values(by='wavelength')
            proc_waves_tar = temp_df_tar['wavelength'].values
            proc_refl_tar = temp_df_tar['reflectance'].values
        
        cont_tar = smooth_continuum(proc_waves_tar, proc_refl_tar)
        cont_rem_tar = proc_refl_tar / (cont_tar + 1e-10)
        
        sam_results = []
        sid_results = []
        sff_results = []
        
        target_log_min = np.log10(target_wavelengths_raw[valid_target_indices].min() if np.any(valid_target_indices) else 1)
        target_log_max = np.log10(target_wavelengths_raw[valid_target_indices].max() if np.any(valid_target_indices) else 1)
        target_log_range = target_log_max - target_log_min
        
        for lib_name in libs:
            lib_waves = df_lib['Wavelength'].values
            lib_refl = df_lib[lib_name].values
            
            valid_lib_indices = ~np.isnan(lib_refl)
            if not np.any(valid_lib_indices): continue
            
            min_common_wavelength = max(target_wavelengths_raw[valid_target_indices].min(), lib_waves[valid_lib_indices].min())
            max_common_wavelength = min(target_wavelengths_raw[valid_target_indices].max(), lib_waves[valid_lib_indices].max())
            
            if min_common_wavelength >= max_common_wavelength: continue
            
            common_log_min_overlap = np.log10(min_common_wavelength)
            common_log_max_overlap = np.log10(max_common_wavelength)
            common_log_overlap_range = common_log_max_overlap - common_log_min_overlap
            
            if target_log_range > 0 and (common_log_overlap_range / target_log_range) < 0.7:
                continue
            
            interp_lib = get_interpolated_reflectance(lib_waves, lib_refl, common_waves.values)
            valid_lib_mask = ~np.isnan(interp_lib)
            if np.sum(valid_lib_mask) < 2: continue
            
            p_w_lib, p_r_lib = get_processed_spectrum(
                common_waves.values[valid_lib_mask],
                interp_lib[valid_lib_mask],
                num_points=100
            )
            if p_w_lib is None: continue
            
            if p_w_lib is not None and p_r_lib is not None:
                temp_df_lib = pd.DataFrame({'wavelength': p_w_lib, 'reflectance': p_r_lib})
                temp_df_lib = temp_df_lib.drop_duplicates(subset=['wavelength']).sort_values(by='wavelength')
                p_w_lib = temp_df_lib['wavelength'].values
                p_r_lib = temp_df_lib['reflectance'].values
            
            common_comp_waves, align_tar, align_lib = align_spectra_for_comparison(
                proc_waves_tar, proc_refl_tar, p_w_lib, p_r_lib
            )
            if align_tar is None: continue
            
            sam = spectral_angle_mapper(align_tar, align_lib)
            if not np.isnan(sam): sam_results.append({'name': lib_name, 'score': sam})
                
            sid = spectral_information_divergence(align_tar, align_lib)
            if not np.isnan(sid): sid_results.append({'name': lib_name, 'score': sid})
                
            if common_comp_waves is not None and len(common_comp_waves) >= 4:
                interp_cr_tar = interp1d(proc_waves_tar, cont_rem_tar, kind='linear', fill_value=np.nan, bounds_error=False)
                align_cr_tar = np.clip(interp_cr_tar(common_comp_waves), 0, None)
                cont_lib_align = smooth_continuum(common_comp_waves, align_lib)
                align_cr_lib = np.clip(align_lib / (cont_lib_align + 1e-10), 0, None)
                valid_rmse = ~np.isnan(align_cr_tar) & ~np.isnan(align_cr_lib)
                if np.any(valid_rmse):
                    rmse = np.sqrt(np.mean((align_cr_tar[valid_rmse] - align_cr_lib[valid_rmse])**2))
                    sff_results.append({'name': lib_name, 'score': rmse})

        sam_results = sorted(sam_results, key=lambda x: x['score'])[:5]
        sid_results = sorted(sid_results, key=lambda x: x['score'])[:5]
        sff_results = sorted(sff_results, key=lambda x: x['score'])[:5]
        
        def generate_plot(best_name):
            best_w = df_lib['Wavelength'].values
            best_r = df_lib[best_name].values
            
            plt.figure(figsize=(10, 4), facecolor='#131313', dpi=200)
            ax = plt.gca()
            ax.set_facecolor('#131313')
            ax.tick_params(colors='#8b90a0')
            for spine in ax.spines.values(): spine.set_color('#414755')
            
            plt.plot(target_wavelengths_raw, target_reflectance_raw, label='Target', color='#ADC6FF', linewidth=1.5)
            plt.plot(best_w, best_r, label='Best Match', color='#00F4FE', linestyle='--', linewidth=1.5)
            
            plt.xscale('log')
            plt.title(f'Spectral Comparison: {best_name}', color='#e5e2e1', fontsize=12)
            plt.xlabel('Wavelength (µm)', color='#8b90a0', fontsize=10)
            plt.ylabel('Reflectance', color='#8b90a0', fontsize=10)
            plt.grid(True, which="both", ls="-", color='#414755', alpha=0.3)
            
            legend = plt.legend()
            plt.setp(legend.get_texts(), color='#e5e2e1')
            legend.get_frame().set_facecolor('#201f1f')
            legend.get_frame().set_edgecolor('#414755')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
            plt.close()
            buf.seek(0)
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        sam_plot = generate_plot(sam_results[0]['name']) if sam_results else None
        sid_plot = generate_plot(sid_results[0]['name']) if sid_results else None
        sff_plot = generate_plot(sff_results[0]['name']) if sff_results else None
        
        return jsonify({
            'sam': sam_results,
            'sid': sid_results,
            'sff': sff_results,
            'sam_plot': sam_plot,
            'sid_plot': sid_plot,
            'sff_plot': sff_plot
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/linear-unmix', methods=['POST'])
def api_linear_unmix():
    locked_minerals_str = request.form.get('locked_minerals')
    locked_minerals = json.loads(locked_minerals_str) if locked_minerals_str else []
    
    ref_file = request.files.get('reflectance')
    wave_file = request.files.get('wavelength')
    
    if not locked_minerals or not ref_file or not wave_file:
        return jsonify({'error': 'Missing inputs'}), 400
        
    try:
        ref_df = read_spectral_data_buffer(ref_file.filename, ref_file.read())
        wave_df = read_spectral_data_buffer(wave_file.filename, wave_file.read())
        
        if len(ref_df) != len(wave_df):
            return jsonify({'error': 'Mismatch in dimensions'}), 400
        
        ref_vals = ref_df['reflectance'].values
        if ref_vals.max() <= 1.01:
            ref_vals = ref_vals * 100
            
        combined_df = pd.DataFrame({
            'wavelength': wave_df['reflectance'].values,
            'reflectance': ref_vals
        })
        
        min_overall = combined_df['wavelength'].min()
        max_overall = combined_df['wavelength'].max()
        
        valid_locks = []
        for m in locked_minerals:
            if m in df_lib.columns:
                valid_locks.append(m)
                min_overall = max(min_overall, df_lib['Wavelength'].min())
                max_overall = min(max_overall, df_lib['Wavelength'].max())
                
        if not valid_locks:
            return jsonify({'error': 'No valid locked minerals found in library'}), 400
            
        common_waves_df = df_lib[(df_lib['Wavelength'] >= min_overall) & (df_lib['Wavelength'] <= max_overall)]['Wavelength']
        
        proc_user_refl = get_interpolated_reflectance(
            combined_df['wavelength'].values,
            combined_df['reflectance'].values,
            common_waves_df.values
        )
        
        endmembers = []
        final_valid_locks = []
        for m in valid_locks:
            m_refl = np.clip(df_lib.loc[common_waves_df.index, m].values, 0, None)
            if not np.all(np.isnan(m_refl)):
                endmembers.append(m_refl)
                final_valid_locks.append(m)
                
        if not final_valid_locks:
             return jsonify({'error': 'No non-NaN valid locked minerals found'}), 400
             
        all_refl = np.vstack([proc_user_refl] + endmembers)
        valid_idx = np.all(np.isfinite(all_refl), axis=0)
        
        b = proc_user_refl[valid_idx]
        A = np.array([spec[valid_idx] for spec in endmembers]).T
        
        x, rnorm = nnls(A, b)
        
        total = np.sum(x)
        percentages = []
        if total > 0:
            pct = (x / total) * 100
            percentages = [{'mineral': final_valid_locks[i], 'percentage': pct[i]} for i in range(len(final_valid_locks))]
            
        recon = np.dot(A, x)
        
        percentages = sorted(percentages, key=lambda p: p['percentage'], reverse=True)
        
        plt.figure(figsize=(10, 4), facecolor='#131313', dpi=200)
        ax = plt.gca()
        ax.set_facecolor('#131313')
        ax.tick_params(colors='#8b90a0')
        for spine in ax.spines.values(): spine.set_color('#414755')
        
        if len(combined_df) < 50:
            plt.plot(combined_df['wavelength'], combined_df['reflectance'], 'o-', label='User Spectrum', color='#ADC6FF', linewidth=1.5)
        else:
            plt.plot(common_waves_df.values[valid_idx], b, label='User Spectrum', color='#ADC6FF', linewidth=2)
            
        plt.plot(common_waves_df.values[valid_idx], recon, label='Reconstructed', color='#00F4FE', linestyle='--', linewidth=2)
        
        plt.xscale('log')
        plt.title('NNLS Spectral Unmixing', color='#e5e2e1', fontsize=12)
        plt.xlabel('Wavelength (µm)', color='#8b90a0', fontsize=10)
        plt.ylabel('Reflectance', color='#8b90a0', fontsize=10)
        plt.grid(True, which="both", ls="-", color='#414755', alpha=0.3)
        legend = plt.legend()
        plt.setp(legend.get_texts(), color='#e5e2e1')
        legend.get_frame().set_facecolor('#201f1f')
        legend.get_frame().set_edgecolor('#414755')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'percentages': percentages,
            'rnorm': rnorm,
            'plot': f'data:image/png;base64,{b64}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
