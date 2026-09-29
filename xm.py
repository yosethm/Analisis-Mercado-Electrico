                label=f"Media móvil ({win})",
            )
            ax.set_title(f"Evolución {freq.lower()} y media móvil")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Precio (COP/kWh)")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            mostrar_analisis(trend_text(res["Precio"], freq))

        elif grafica.startswith("2."):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(res["Precio"].dropna(), bins=min(30, max(8, int(np.sqrt(len(res))))), alpha=0.85)
            ax.axvline(res["Precio"].mean(), linestyle="--", linewidth=1.5, label="Media")
            ax.axvline(res["Precio"].median(), linestyle=":", linewidth=1.8, label="Mediana")
            ax.set_title("Distribución de precios")
            ax.set_xlabel("Precio (COP/kWh)")
            ax.set_ylabel("Frecuencia")
            ax.grid(axis="y", alpha=0.22)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            mostrar_analisis(dist_text(res["Precio"]))

        elif grafica.startswith("3."):
            df_box = df_vis.copy()
            df_box["MesN"] = df_box["Fecha"].dt.month
            meses_presentes = sorted(df_box["MesN"].dropna().unique().tolist())
            etiquetas = [calendar.month_name[m] for m in meses_presentes]
            series = [df_box.loc[df_box["MesN"] == m, "Valor"].dropna().to_numpy() for m in meses_presentes]

            fig, ax = plt.subplots(figsize=(14, 5))
            if series:
                ax.boxplot(series, labels=etiquetas, showfliers=True)
            ax.set_title("Distribución de precios por mes")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Precio (COP/kWh)")
            ax.tick_params(axis="x", rotation=28)
            ax.grid(axis="y", alpha=0.22)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            df_box["Mes"] = df_box["MesN"].map(lambda m: calendar.month_name[m])
            mostrar_analisis(box_text(df_box))

        elif grafica.startswith("4."):
            df_hm = df_vis.copy()
            df_hm["Año"] = df_hm["Fecha"].dt.year
            df_hm["MesN"] = df_hm["Fecha"].dt.month
            piv = (
                df_hm.pivot_table(index="Año", columns="MesN", values="Valor", aggfunc="mean")
                .reindex(columns=range(1, 13))
            )
            piv.columns = [calendar.month_abbr[m] for m in piv.columns]

            fig, ax = plt.subplots(figsize=(12, 5.5))
            masked = np.ma.masked_invalid(piv.to_numpy(dtype=float))
            im = ax.imshow(masked, aspect="auto", interpolation="nearest")
            ax.set_title("Promedio de precios por Año y Mes")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Año")
            ax.set_xticks(np.arange(len(piv.columns)), labels=piv.columns)
            ax.set_yticks(np.arange(len(piv.index)), labels=piv.index)
            fig.colorbar(im, ax=ax, label="COP/kWh")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            mostrar_analisis(heat_text(piv))

        elif grafica.startswith("5."):
            df_lag = df_vis[["Valor"]].copy()
            df_lag["Valor_lag1"] = df_lag["Valor"].shift(1)
            df_lag = df_lag.dropna()
            corr = df_lag["Valor_lag1"].corr(df_lag["Valor"]) if len(df_lag) > 1 else np.nan

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(df_lag["Valor_lag1"], df_lag["Valor"], s=24, alpha=0.55)
            if len(df_lag) >= 2:
                coef = np.polyfit(df_lag["Valor_lag1"], df_lag["Valor"], 1)
                x_line = np.linspace(df_lag["Valor_lag1"].min(), df_lag["Valor_lag1"].max(), 100)
                ax.plot(x_line, coef[0] * x_line + coef[1], linewidth=2)
            ax.set_title("Relación precio actual vs. periodo anterior")
            ax.set_xlabel("Precio periodo anterior (COP/kWh)")
            ax.set_ylabel("Precio actual (COP/kWh)")
            ax.grid(alpha=0.22)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            mostrar_analisis(pers_text(corr))

        else:
            ult_12m = df_vis[df_vis["Fecha"] >= df_vis["Fecha"].max() - pd.Timedelta(days=365)]
            if ult_12m.empty:
                st.info("No hay datos suficientes en los últimos 12 meses.")
            else:
                top_max = (
                    ult_12m.nlargest(10, "Valor")[["Fecha", "Valor"]]
                    .rename(columns={"Valor": "Precio"})
                    .reset_index(drop=True)
                )
                top_min = (
                    ult_12m.nsmallest(10, "Valor")[["Fecha", "Valor"]]
                    .rename(columns={"Valor": "Precio"})
                    .reset_index(drop=True)
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🔺 Máximos")
                    st.dataframe(top_max, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("#### 🔻 Mínimos")
                    st.dataframe(top_min, use_container_width=True, hide_index=True)

                amplitud = ult_12m["Valor"].max() - ult_12m["Valor"].min()
                mostrar_analisis(
                    f"Amplitud del último año: **{amplitud:.2f} COP/kWh**. "
                    f"Último valor disponible: **{df_vis['Valor'].iloc[-1]:.2f} COP/kWh**."
                )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer">
        <p>⚡ <b>Yoseth Mosquera</b> · Universidad de Antioquia</p>
        <p>📊 Fuente de datos: <b>SIMEM</b></p>
        <p class="muted">© {datetime.now().year} · Aplicación optimizada para consulta y visualización</p>
