def make_template(
    variables,
):
    """
    Make the digital effects table template for the case where uncertainties are combined and only the total uncertainty is returned.

    :param dims: list of dimensions
    :type dims: list
    :param u_xvar_ref: reference uncertainty component that is used to populate repeated dims
    :type u_xvar_ref: xarray.Variable
    :return: measurand digital effects table template to be used by obsarray
    :rtype: dict
    """
    template = {}

    for i in range(self.output_vars):
        err_corr, custom_err_corr = self._set_errcorr_shape(
            dims[i],
            dim_sizes,
            "err_corr_tot_" + self.yvariable[i],
            str_repeat_noncorr_dims=str_repeat_noncorr_dims,
            str_corr_dims=str_corr_dims[i],
            repeat_dim_err_corr=repeat_dim_err_corrs,
        )

        if store_unc_percent:
            units = "%"
        else:
            units = self.yunit[i]

        template[self.yvariable[i]] = {
            "dtype": np.float32,
            "dim": dims[i],
            "attributes": {
                "units": self.yunit[i],
                "unc_comps": [
                    self._make_ucomp_name(
                        self.yvariable[i],
                        "tot",
                        store_unc_percent=store_unc_percent,
                    )
                ],
            },
        }
        template[
            self._make_ucomp_name(
                self.yvariable[i], "tot", store_unc_percent=store_unc_percent
            )
        ] = {
            "dtype": np.float32,
            "dim": dims[i],
            "attributes": {"units": units, "err_corr": err_corr},
        }

        if custom_err_corr is not None:
            for key in custom_err_corr.keys():
                if "err_corr_tot_" + self.yvariable[i] in key:
                    template[key] = custom_err_corr[key]

    # if self.output_vars>1:
    #     template["corr_between_vars"]={
    #         "dtype": np.float32,
    #         "dim": dims,
    #         "attributes": {"units": units, "err_corr": err_corr},
    #     }

    return template
