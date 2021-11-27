PRO test
        jsamr2cell, cell, dir='/storage5/FORNAX/KISTI_OUTPUT/l10006/', snapnum=171, domlist=[420, 421L], /pointer
        xr      = [0.51068011,      0.51228593]*1.d
        yr      = [0.49339595,      0.49500177]*1.d
        rd_info, info, file='/storage5/FORNAX/KISTI_OUTPUT/l10006/output_00171/info_00171.txt'
        a=js_gasmap(cell, info, xr=xr, yr=yr, n_pix=1000L, amrtype='Den', minlev=8L, maxlev=21L, proj='xy', n_thread=1L)

END
