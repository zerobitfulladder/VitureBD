"""menu.py — a small keyboard-driven settings menu (no GL, no mouse).

Settings are grouped into Tabs (switched with Tab / Shift+Tab). Within a tab,
items are either a Section (a non-selectable subheader) or a Setting (a typed
value: float / int / bool / enum) with get/set closures over the app's state.
Settings may carry a `visible` predicate so context-dependent knobs (e.g. JBU vs
guided-filter options) appear only when relevant, and a `help` one-liner shown in
the info pane. Up/Down move over the visible editable rows of the current tab;
Left/Right adjust; Tab/Shift+Tab change tab; the app dismisses on Enter.
"""


class Section:
    """A non-selectable subheader that groups the settings beneath it."""
    is_section = True

    def __init__(self, label):
        self.label = label


class Setting:
    is_section = False

    def __init__(self, label, kind, get, set, *, step=1, lo=None, hi=None,
                 options=None, fmt=None, editable=True, visible=None, help=""):
        self.label = label
        self.kind = kind            # 'num' | 'bool' | 'enum'
        self._get, self._set = get, set
        self.step, self.lo, self.hi = step, lo, hi
        self.options = options or []
        self._fmt = fmt
        self.editable = editable
        self._visible = visible     # None => always visible
        self.help = help

    @property
    def visible(self):
        return self._visible is None or bool(self._visible())

    @property
    def value(self):
        return self._get()

    def display(self):
        v = self._get()
        if self._fmt:
            return self._fmt(v)
        if self.kind == "bool":
            return "ON" if v else "OFF"
        if self.kind == "enum":
            return str(v)
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    def adjust(self, direction):
        """direction: -1 (Left) or +1 (Right). Returns True if the value changed."""
        if not self.editable:
            return False
        v = self._get()
        if self.kind == "bool":
            nv = not v
        elif self.kind == "enum":
            i = self.options.index(v) if v in self.options else 0
            nv = self.options[(i + direction) % len(self.options)]
        else:  # num
            nv = v + direction * self.step
            if self.lo is not None:
                nv = max(self.lo, nv)
            if self.hi is not None:
                nv = min(self.hi, nv)
            if isinstance(v, int) and not isinstance(self.step, float):
                nv = int(round(nv))
        if nv == v:
            return False
        self._set(nv)
        return True


class Tab:
    def __init__(self, name, items):
        self.name = name
        self.items = items          # Section | Setting, in display order


class Menu:
    def __init__(self, tabs):
        self.tabs = tabs
        self.t = 0                  # current tab index
        self._sel = 0               # selection within the current tab's nav list

    @property
    def tab(self):
        return self.tabs[self.t]

    def _nav(self):
        """Visible, editable settings in the current tab — where the cursor lands."""
        return [it for it in self.tab.items
                if not it.is_section and it.editable and it.visible]

    @property
    def current(self):
        nav = self._nav()
        if not nav:
            return None
        self._sel = min(self._sel, len(nav) - 1)   # clamp if rows hid out
        return nav[self._sel]

    def next_tab(self):
        self.t = (self.t + 1) % len(self.tabs)
        self._sel = 0

    def prev_tab(self):
        self.t = (self.t - 1) % len(self.tabs)
        self._sel = 0

    def up(self):
        nav = self._nav()
        if nav:
            self._sel = (min(self._sel, len(nav) - 1) - 1) % len(nav)

    def down(self):
        nav = self._nav()
        if nav:
            self._sel = (min(self._sel, len(nav) - 1) + 1) % len(nav)

    def left(self):
        c = self.current
        return c.adjust(-1) if c else False

    def right(self):
        c = self.current
        return c.adjust(+1) if c else False

    def tab_bar(self):
        """(names, current_index) for the renderer's tab strip."""
        return [t.name for t in self.tabs], self.t

    def rows(self):
        """Rows of the CURRENT tab only (sections + visible settings)."""
        cur = self.current
        out = []
        for it in self.tab.items:
            if it.is_section:
                out.append({"section": True, "label": it.label})
            elif it.visible:
                out.append({"section": False, "label": it.label, "value": it.display(),
                            "kind": it.kind, "editable": it.editable,
                            "selected": (it is cur), "help": it.help})
        return out
