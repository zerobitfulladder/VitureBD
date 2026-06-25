"""picker.py — cyberpunk tkinter dialog to choose a display profile before launch.

Shows detected outputs + glasses status (xrandr + USB), a profile list, and a
LAUNCH/APPLY button. Returns the chosen profile name, or None if cancelled.
Palette mirrors ~/Scripts/palette_cyberpunk.py (Night City neon).
"""
import tkinter as tk
import tkinter.font as tkfont

C = {
    "bg": "#0a0418", "panel": "#0d0820", "gutter": "#15102a", "surface": "#1e1838",
    "muted": "#5a4a78", "fg": "#a0a8c8", "fg_hi": "#c8d0e8",
    "magenta": "#ff2a6d", "magenta_mid": "#d1004f",
    "cyan": "#05d9e8", "cyan_mid": "#05a9c4", "yellow": "#fcee0c",
    "lime": "#ccff00", "violet": "#b967ff", "error": "#ff003c",
}


def _mono(size, weight="normal"):
    for fam in ("JetBrains Mono", "DejaVu Sans Mono", "Consolas", "monospace"):
        try:
            return tkfont.Font(family=fam, size=size, weight=weight)
        except tk.TclError:
            continue
    return tkfont.Font(size=size, weight=weight)


def choose(outs, glasses, profiles, auto=None):
    """auto=<name> (headless self-test): build the UI hidden, confirm it, return."""
    root = tk.Tk()
    root.title("BRAINDANCE")
    root.configure(bg=C["bg"])
    root.resizable(False, False)
    try:
        root.attributes("-type", "dialog")
    except tk.TclError:
        pass

    f_title = _mono(22, "bold")
    f_sub = _mono(9)
    f_h = _mono(10, "bold")
    f_body = _mono(11)
    f_small = _mono(9)
    f_btn = _mono(12, "bold")

    pad = {"padx": 22}
    result = {"name": None}
    names = sorted(profiles, key=lambda n: profiles[n].get("order", 99))
    sel = tk.StringVar()

    # default selection: braindance if glasses are 3D-ready, else first non-run profile
    if glasses.get("can_3d"):
        sel.set("braindance" if "braindance" in profiles else names[0])
    else:
        sel.set(next((n for n in names if not profiles[n].get("run")), names[0]))

    # ── header ──────────────────────────────────────────────────────────
    head = tk.Frame(root, bg=C["bg"])
    head.pack(fill="x", pady=(18, 4), **pad)
    title = tk.Frame(head, bg=C["bg"])
    title.pack(anchor="w", fill="x")
    tk.Label(title, text="BRAINDANCE", font=f_title, fg=C["magenta"], bg=C["bg"]).pack(side="left")
    tk.Label(title, text="  v3.7-NC", font=f_sub, fg=C["yellow"], bg=C["bg"]).pack(side="left", pady=(10, 0))
    tk.Label(head, text="// jack in — select a display feed", font=f_sub, fg=C["cyan_mid"],
             bg=C["bg"]).pack(anchor="w")
    tk.Frame(root, bg=C["magenta_mid"], height=2).pack(fill="x", pady=(8, 0), **pad)

    # ── outputs panel ───────────────────────────────────────────────────
    op = tk.Frame(root, bg=C["panel"])
    op.pack(fill="x", pady=(14, 6), **pad)
    tk.Label(op, text="OUTPUTS", font=f_h, fg=C["violet"], bg=C["panel"]).pack(
        anchor="w", padx=12, pady=(8, 2))
    for name in sorted(outs):
        o = outs[name]
        if not o["connected"] and not o["active"]:
            continue
        if o["active"]:
            dot, dotc = "●", C["lime"]
            state = o["mode"] + (" *" if o["primary"] else "")
            statec = C["fg_hi"]
        elif o["connected"]:
            dot, dotc, state, statec = "○", C["yellow"], "connected / off", C["yellow"]
        else:
            dot, dotc, state, statec = "○", C["muted"], "off", C["muted"]
        row = tk.Frame(op, bg=C["panel"])
        row.pack(fill="x", padx=12, pady=1)
        tk.Label(row, text=dot, font=f_small, fg=dotc, bg=C["panel"]).pack(side="left")
        tk.Label(row, text=f" {name:<10s}", font=f_body, fg=C["cyan"],
                 bg=C["panel"]).pack(side="left")
        tk.Label(row, text=state, font=f_small, fg=statec, bg=C["panel"]).pack(side="left")
    # glasses status line
    gcol = (C["lime"] if glasses["in_3d"] else
            C["yellow"] if (glasses["usb"] or glasses["connected"]) else C["error"])
    gl = tk.Frame(op, bg=C["panel"])
    gl.pack(fill="x", padx=12, pady=(6, 10))
    tk.Label(gl, text="◈ GLASSES  ", font=f_small, fg=C["magenta"],
             bg=C["panel"]).pack(side="left")
    tk.Label(gl, text=glasses["status"], font=f_small, fg=gcol, bg=C["panel"]).pack(side="left")

    # ── profile list ────────────────────────────────────────────────────
    tk.Label(root, text="PROFILE", font=f_h, fg=C["violet"], bg=C["bg"]).pack(
        anchor="w", pady=(8, 4), **pad)
    rows = {}

    def refresh():
        for n, (frame, ind, lbl, desc) in rows.items():
            on = (sel.get() == n)
            frame.configure(bg=C["surface"] if on else C["panel"],
                            highlightbackground=C["magenta"] if on else C["gutter"])
            ind.configure(text="▸" if on else " ",
                          fg=C["magenta"] if on else C["muted"],
                          bg=C["surface"] if on else C["panel"])
            lbl.configure(fg=C["fg_hi"] if on else C["fg"],
                          bg=C["surface"] if on else C["panel"])
            desc.configure(fg=C["fg"] if on else C["muted"],
                           bg=C["surface"] if on else C["panel"])
        ok_btn.configure(text=("LAUNCH ▸" if profiles[sel.get()].get("run")
                               else "APPLY ▸"))

    def pick(n):
        sel.set(n)
        refresh()

    for n in names:
        p = profiles[n]
        fr = tk.Frame(root, bg=C["panel"], highlightthickness=1,
                      highlightbackground=C["gutter"])
        fr.pack(fill="x", pady=3, **pad)
        top = tk.Frame(fr, bg=C["panel"])
        top.pack(fill="x", padx=10, pady=(6, 0))
        ind = tk.Label(top, text=" ", font=f_body, fg=C["muted"], bg=C["panel"])
        ind.pack(side="left")
        lbl = tk.Label(top, text=" " + p.get("label", n), font=f_body, fg=C["fg"],
                       bg=C["panel"])
        lbl.pack(side="left")
        if p.get("run"):
            tk.Label(top, text="  3D", font=f_small, fg=C["lime"], bg=C["panel"]).pack(side="left")
        desc = tk.Label(fr, text=p.get("desc", ""), font=f_small, fg=C["muted"],
                        bg=C["panel"], anchor="w", justify="left", wraplength=480)
        desc.pack(fill="x", padx=28, pady=(0, 7))
        rows[n] = (fr, ind, lbl, desc)
        for w in (fr, top, ind, lbl, desc):
            w.bind("<Button-1>", lambda e, nm=n: pick(nm))

    # ── buttons ─────────────────────────────────────────────────────────
    def do_ok(*_):
        result["name"] = sel.get()
        root.destroy()

    def do_cancel(*_):
        result["name"] = None
        root.destroy()

    bar = tk.Frame(root, bg=C["bg"])
    bar.pack(fill="x", pady=(14, 18), **pad)
    cancel_btn = tk.Button(bar, text="CANCEL", font=f_btn, command=do_cancel,
                           fg=C["muted"], bg=C["gutter"], activeforeground=C["error"],
                           activebackground=C["surface"], relief="flat", bd=0,
                           padx=18, pady=8, cursor="hand2")
    cancel_btn.pack(side="left")
    ok_btn = tk.Button(bar, text="LAUNCH ▸", font=f_btn, command=do_ok,
                       fg=C["bg"], bg=C["magenta"], activeforeground=C["bg"],
                       activebackground=C["magenta_mid"], relief="flat", bd=0,
                       padx=26, pady=8, cursor="hand2")
    ok_btn.pack(side="right")

    refresh()
    root.bind("<Return>", do_ok)
    root.bind("<Escape>", do_cancel)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f"+{x}+{y}")
    ok_btn.focus_set()
    if auto is not None:
        root.withdraw()                       # headless self-test: don't show
        if auto in profiles:
            sel.set(auto)
        refresh()
        root.after(60, do_ok)
    root.mainloop()
    return result["name"]
