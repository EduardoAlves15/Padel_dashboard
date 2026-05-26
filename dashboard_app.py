import streamlit as st
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from padel_pipeline import (
    APP_TITLE,
    DEFAULT_ACTION_MODEL,
    DEFAULT_BALL_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_MODEL,
    ball_metrics_from_any_schema,
    court_figure,
    validate_calibration,
        get_calibration,
        auto_fix_calibration_permutations,
        set_calibration,
    load_action_model,
    load_ball_model,
    load_pose_model,
    persist_uploaded_video,
    preview_image,
    refine_ball_stage,
    run_detection_stage,
    analyze_stage,
    save_json,
    reprojection_error_2view,
    get_geometry_matrices,
)


def persist_uploaded_model(uploaded_file, suffix: str) -> str:
    if uploaded_file is None:
        return ""
    temp_dir = Path(tempfile.mkdtemp(prefix="padel_model_"))
    target = temp_dir / f"{uploaded_file.name.rsplit('.', 1)[0]}{suffix}"
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return str(target)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Carrega dois vídeos, escolhe os modelos e corre a análise numa única interface.")

    with st.sidebar:
        with st.expander("Como usar a app", expanded=True):
            st.markdown(
                """
                1. Carrega dois vídeos do mesmo ponto/jogo, já sincronizados.
                2. Faz upload dos modelos da bola, pose e ações diretamente na sidebar.
                3. Se necessário, valida a calibração antes de correr a análise.
                4. Clica em **Executar análise completa**.
                5. Vê os resultados nos separadores de resumo, tabelas, gráficos e JSON.
                """
            )
        st.header("Configuração")
        st.subheader("Modelos")
        ball_model_upload = st.file_uploader("Upload modelo bola (.pt)", type=["pt"], key="ball_model_upload")
        pose_model_upload = st.file_uploader("Upload modelo pose (.pt)", type=["pt"], key="pose_model_upload")
        action_model_upload = st.file_uploader("Upload modelo ações (.h5)", type=["h5"], key="action_model_upload")
        video_main_upload = st.file_uploader("Vídeo principal", type=["mp4", "avi", "mov", "mkv"], key="main_video")
        video_second_upload = st.file_uploader("Segundo vídeo", type=["mp4", "avi", "mov", "mkv"], key="second_video")
        st.number_input("Máximo de frames", min_value=10, max_value=5000, value=150, step=10, key="max_frames")
        st.number_input("Threshold ação", min_value=0.1, max_value=0.99, value=0.70, step=0.05, key="action_threshold")
        st.number_input("Confiança bola", min_value=0.01, max_value=0.99, value=0.15, step=0.01, key="ball_conf")
        st.number_input("Image size", min_value=640, max_value=1920, value=1280, step=64, key="imgsz")
        output_dir = st.text_input("Pasta de saída", value=str(DEFAULT_OUTPUT_DIR))
        run_button = st.button("Executar análise completa", type="primary")
        calib_button = st.button("Validar calibração")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Vídeo principal")
        if video_main_upload is not None:
            st.video(video_main_upload)
            image = preview_image(video_main_upload)
            if image is not None:
                st.image(image, caption="Pré-visualização do principal", use_container_width=True)
        else:
            st.info("Carrega um vídeo principal para começar.")
    with col_right:
        st.subheader("Segundo vídeo")
        if video_second_upload is not None:
            st.video(video_second_upload)
            image = preview_image(video_second_upload)
            if image is not None:
                st.image(image, caption="Pré-visualização do segundo", use_container_width=True)
        else:
            st.info("Carrega o segundo vídeo para a triangulação estéreo.")

    if calib_button:
        st.info("Validando calibração...")
        try:
            res = validate_calibration()
            rmse = res.get('rmse', float('nan'))
            if np.isfinite(rmse):
                st.success(f"RMSE: {rmse:.3f} m")
            else:
                st.warning("RMSE não disponível (mapeamento não finito)")

            # display a small table of mappings
            rows = []
            from padel_pipeline import PTS_MAIN_2D, METROS_REAIS_2D
            for img_pt, mapped, expect in zip(PTS_MAIN_2D, res['mapped'], res['expected']):
                rows.append({
                    'image_pt': f"({float(img_pt[0]):.1f}, {float(img_pt[1]):.1f})",
                    'mapped_m': f"({None if mapped[0] is None else f'{mapped[0]:.3f}'}, {None if mapped[1] is None else f'{mapped[1]:.3f}'})",
                    'expected_m': f"({float(expect[0]):.3f}, {float(expect[1]):.3f})",
                })
            st.table(rows)
            if np.isfinite(rmse) and rmse > 0.5:
                st.warning("RMSE elevado — verifica a ordem dos pontos de calibração ou unidades (metros).")
        except Exception as e:
            st.error(f"Erro ao validar calibração: {e}")

        # show current calibration and offer auto-fix
        try:
            calib = get_calibration()
            st.subheader('Calibração atual')
            st.json(calib)
            if st.button('Auto-corrigir ordem dos METROS_REAIS'):
                with st.spinner('A testar permutações...'):
                    result = auto_fix_calibration_permutations()
                if result['improved']:
                    st.success(f"Auto-corrigido — novo RMSE {result['best_rmse']:.4f} m")
                    st.json(get_calibration())
                    save_path = Path('padel_calibration.json')
                    save_path.write_text(json.dumps(get_calibration(), indent=2), encoding='utf-8')
                    st.info(f"Calibração guardada em {save_path}")
                else:
                    st.info('Nenhuma permutação melhor encontrada.')
        except Exception:
            pass

        # Import / Export calibration
        st.markdown("### Importar/Exportar calibração")
        cal_file = st.file_uploader("Importar calibração (JSON)", type=["json"], key="import_calib")
        if cal_file is not None:
            try:
                loaded = json.load(cal_file)
                pts_main = loaded.get("PTS_MAIN_2D") or loaded.get("pts_main")
                pts_second = loaded.get("PTS_SECOND_2D") or loaded.get("pts_second")
                metros = loaded.get("METROS_REAIS_2D") or loaded.get("metros_reais")
                set_calibration(pts_main=pts_main, pts_second=pts_second, metros=metros)
                st.success("Calibração importada e aplicada.")
            except Exception as e:
                st.error(f"Erro ao importar calibração: {e}")

        # Manual edit of METROS_REAIS_2D
        try:
            from padel_pipeline import METROS_REAIS_2D, PTS_MAIN_2D
            metros_current = METROS_REAIS_2D.tolist()
            with st.form("edit_calib_form"):
                st.write("Editar METROS_REAIS_2D (metros) — pontos de canto do campo")
                new_metros = []
                for i, pt in enumerate(metros_current):
                    c1, c2 = st.columns(2)
                    x = c1.number_input(f"Ponto {i+1} X (m)", value=float(pt[0]), step=0.1, format="%.3f", key=f"metx{i}")
                    y = c2.number_input(f"Ponto {i+1} Y (m)", value=float(pt[1]), step=0.1, format="%.3f", key=f"mety{i}")
                    new_metros.append([x, y])
                submitted = st.form_submit_button("Salvar calibração manual")
                if submitted:
                    try:
                        set_calibration(pts_main=PTS_MAIN_2D, metros=new_metros)
                        save_path = Path('padel_calibration.json')
                        save_path.write_text(json.dumps(get_calibration(), indent=2), encoding='utf-8')
                        st.success("Calibração atualizada e guardada em padel_calibration.json")
                    except Exception as e:
                        st.error(f"Erro a guardar calibração: {e}")
        except Exception:
            pass

        if st.button("Exportar calibração atual"):
            try:
                save_path = Path("padel_calibration.json")
                save_path.write_text(json.dumps(get_calibration(), indent=2), encoding='utf-8')
                st.success(f"Calibração exportada para {save_path}")
            except Exception as e:
                st.error(f"Erro a exportar calibração: {e}")

        # Geometry issues viewer
        st.markdown("### Problemas de geometria")
        tmp_path = Path(tempfile.gettempdir()) / "padel_geometry_issues.json"
        if tmp_path.exists():
            try:
                txt = tmp_path.read_text(encoding='utf-8')
                issues = json.loads(txt)
                st.write(f"Registos: {len(issues)}")
                sel = st.selectbox('Seleciona registo para visualizar', list(range(min(200, len(issues)))), format_func=lambda i: f"{i} - frame {issues[i]['frame']} ({issues[i].get('type')})")
                st.dataframe(issues[:200])
                issue = issues[sel]
                st.markdown('**Visualização do registo selecionado**')
                try:
                    # prepare plot: show camera1 and camera2 coordinate spaces
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    p1 = issue.get('p1')
                    p2 = issue.get('p2')
                    tri = issue.get('triangulated')
                    P1, P2, H = get_geometry_matrices()
                    if tri is not None and p1 is not None and p2 is not None:
                        # compute reprojections from triangulated 3D if finite
                        if all(x is not None for x in tri) and np.isfinite(np.array(tri)).all():
                            e1, e2 = reprojection_error_2view(tri, p1, p2)
                            # project using P1/P2 for display
                            X = np.array([tri[0], tri[1], tri[2], 1.0], dtype=np.float32)
                            proj1 = P1 @ X
                            proj2 = P2 @ X
                            u1 = proj1[0] / proj1[2]
                            v1 = proj1[1] / proj1[2]
                            u2 = proj2[0] / proj2[2]
                            v2 = proj2[1] / proj2[2]
                        else:
                            u1 = v1 = u2 = v2 = None
                            e1 = e2 = None
                    else:
                        u1 = v1 = u2 = v2 = None
                        e1 = e2 = None

                    # left plot: camera1
                    ax = axes[0]
                    ax.set_title('Camera 1')
                    ax.scatter([p1[0]], [p1[1]], c='red', label='detected p1')
                    if u1 is not None:
                        ax.scatter([u1], [v1], c='green', label=f'reprojected (err={e1:.1f}px)')
                    ax.invert_yaxis()
                    ax.legend()

                    # right plot: camera2
                    ax2 = axes[1]
                    ax2.set_title('Camera 2')
                    ax2.scatter([p2[0]], [p2[1]], c='red', label='detected p2')
                    if u2 is not None:
                        ax2.scatter([u2], [v2], c='green', label=f'reprojected (err={e2:.1f}px)')
                    ax2.invert_yaxis()
                    ax2.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erro a gerar visualização: {e}")
                st.download_button("Baixar problemas de geometria (JSON)", txt, file_name="padel_geometry_issues.json", mime="application/json")
            except Exception as e:
                st.error(f"Erro a ler ficheiro de problemas: {e}")
        else:
            st.info("Nenhum ficheiro padel_geometry_issues.json no TEMP.")

    if run_button:
        if video_main_upload is None or video_second_upload is None:
            st.error("Precisas de carregar os dois vídeos.")
            return

        ball_model_path = persist_uploaded_model(ball_model_upload, ".pt") or st.session_state.ball_model_path
        pose_model_path = persist_uploaded_model(pose_model_upload, ".pt") or st.session_state.pose_model_path
        action_model_path = persist_uploaded_model(action_model_upload, ".h5") or st.session_state.action_model_path

        video_main_path = persist_uploaded_video(video_main_upload, Path(video_main_upload.name).suffix)
        video_second_path = persist_uploaded_video(video_second_upload, Path(video_second_upload.name).suffix)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0)
        status = st.empty()

        status.write("A correr deteção e tracking...")
        raw_data = run_detection_stage(
            video_main_path=video_main_path,
            video_second_path=video_second_path,
            ball_model_path=ball_model_path,
            pose_model_path=pose_model_path,
            max_frames=int(st.session_state.max_frames),
            ball_conf=float(st.session_state.ball_conf),
            imgsz=int(st.session_state.imgsz),
        )
        progress.progress(30)
        save_json(output_root / "data_final_tese.json", raw_data)

        status.write("A refinar coordenadas...")
        refined_data = refine_ball_stage(raw_data, max_speed_kmh=165, window=5)
        progress.progress(60)
        save_json(output_root / "COORDENADAS_ESTAVEIS_3D.json", refined_data)

        status.write("A calcular análise final...")
        final_report = analyze_stage(
            refined_data,
            action_model_path=action_model_path,
            threshold=float(st.session_state.action_threshold),
            fps=50,
        )
        progress.progress(85)
        save_json(output_root / "analise_final_completa.json", final_report)

        df_metrics = ball_metrics_from_any_schema(final_report)
        player_rows = []
        for frame in final_report:
            for player in frame.get("players", []):
                pos = player.get("pos_m") or player.get("pos")
                if pos is None:
                    continue
                x_val, y_val = pos[0], pos[1]
                if x_val is None or y_val is None:
                    continue
                if np.isnan(float(x_val)) or np.isnan(float(y_val)):
                    continue
                player_rows.append({
                    "frame": int(frame.get("frame", 0)),
                    "player_id": int(player.get("id", -1)),
                    "x": float(x_val),
                    "y": float(y_val),
                    "action": player.get("action", ""),
                    "confidence": float(player.get("confidence", 0.0)),
                })
        df_players = pd.DataFrame(player_rows)
        if not df_metrics.empty:
            df_metrics.to_csv(output_root / "dados_limpos_3d.csv", index=False)

        progress.progress(100)
        status.success("Análise concluída.")

        st.success(f"Resultados guardados em {output_root}")

        tab_summary, tab_tables, tab_plots, tab_raw = st.tabs(["Resumo", "Tabelas", "Gráficos", "JSON"])

        with tab_summary:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames processados", len(raw_data))
            c2.metric("Frames analisados", len(final_report))
            max_z = float(df_metrics["z"].max()) if not df_metrics.empty else float("nan")
            c3.metric("Altura máxima da bola", f"{max_z:.2f} m" if not np.isnan(max_z) else "n/a")

        with tab_tables:
            if not df_metrics.empty:
                st.dataframe(df_metrics.head(50), use_container_width=True)
            else:
                st.warning("Sem métricas da bola nesta run. A mostrar posições dos jogadores como fallback.")
            if not df_players.empty:
                st.subheader("Posições dos jogadores")
                st.dataframe(df_players.head(100), use_container_width=True)
            else:
                st.info("Sem posições de jogadores válidas para mostrar.")

        with tab_plots:
            if not df_metrics.empty:
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df_metrics["frame"], df_metrics["z"], color="#1d4ed8", lw=2)
                ax1.set_title("Altura da bola")
                ax1.set_xlabel("Frame")
                ax1.set_ylabel("Z (m)")
                ax1.grid(alpha=0.2)
                st.pyplot(fig1, clear_figure=True)

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(df_metrics["frame"], df_metrics["speed_kmh"], color="#16a34a", lw=2)
                ax2.set_title("Velocidade da bola")
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("km/h")
                ax2.grid(alpha=0.2)
                st.pyplot(fig2, clear_figure=True)

            if final_report:
                fig_court = court_figure(final_report)
                st.pyplot(fig_court, clear_figure=True)
            elif not df_players.empty:
                fig_court, ax = plt.subplots(figsize=(7, 10))
                ax.plot([0, 10, 10, 0, 0], [0, 0, 20, 20, 0], color="black", lw=2)
                ax.scatter(df_players["x"], df_players["y"], c="#2a9d8f", s=18, alpha=0.6)
                ax.set_xlim(-1, 11)
                ax.set_ylim(-1, 21)
                ax.set_aspect("equal")
                ax.set_title("Posições dos jogadores (fallback)")
                ax.set_xlabel("Largura (m)")
                ax.set_ylabel("Comprimento (m)")
                ax.grid(alpha=0.15)
                st.pyplot(fig_court, clear_figure=True)

        with tab_raw:
            st.subheader("data_final_tese.json")
            st.json(raw_data[:5])
            st.subheader("COORDENADAS_ESTAVEIS_3D.json")
            st.json(refined_data[:5])
            st.subheader("analise_final_completa.json")
            st.json(final_report[:5])


if __name__ == "__main__":
    main()