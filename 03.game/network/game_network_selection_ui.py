"""Network-only screen for choosing a team preset or a custom roster."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from character_stats import all_names
from party_presets import all_preset_names, get_preset

ORIGINAL_TEAM_OPTION = "Original (custom roster)"


class NetworkRosterSelectionWindow:
    """Choose one player's team, roster, and server endpoint."""

    def __init__(self, on_confirm, default_host: str, default_port: int,
                 default_team: str = "auto"):
        self.on_confirm = on_confirm
        self.root = tk.Tk()
        self.root.title("Network Team Selection")
        self.root.geometry("760x650")
        self.root.minsize(680, 560)
        self.roster: list[str] = []
        self.preset_active = False

        self.team_var = tk.StringVar(self.root, value=default_team)
        self.host_var = tk.StringVar(self.root, value=default_host)
        self.port_var = tk.StringVar(self.root, value=str(default_port))
        self.team_name_var = tk.StringVar(self.root, value="")
        self.preset_var = tk.StringVar(self.root, value=ORIGINAL_TEAM_OPTION)
        self.spike_var = tk.StringVar(self.root, value="")
        self.igl_var = tk.StringVar(self.root, value="")
        self.status_var = tk.StringVar(self.root, value="Select a team preset or choose five characters.")

        endpoint = ttk.LabelFrame(self.root, text="Match and team", padding=8)
        endpoint.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(endpoint, text="Match team").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(endpoint, self.team_var, default_team, "auto", "A", "D").grid(
            row=0, column=1, sticky="ew", padx=(6, 16),
        )
        ttk.Label(endpoint, text="Server IP / host").grid(row=0, column=2, sticky="w")
        ttk.Entry(endpoint, textvariable=self.host_var, width=24).grid(
            row=0, column=3, sticky="ew", padx=(6, 12),
        )
        ttk.Label(endpoint, text="Port").grid(row=0, column=4, sticky="w")
        ttk.Entry(endpoint, textvariable=self.port_var, width=8).grid(
            row=0, column=5, sticky="ew", padx=(6, 0),
        )
        ttk.Label(endpoint, text="Team preset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        preset_names = all_preset_names()
        ttk.OptionMenu(endpoint, self.preset_var, ORIGINAL_TEAM_OPTION,
                       ORIGINAL_TEAM_OPTION, *preset_names,
                       command=self.select_preset).grid(
            row=1, column=1, columnspan=5, sticky="ew", padx=(6, 0), pady=(8, 0),
        )
        ttk.Label(endpoint, text="Team name (optional)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(endpoint, textvariable=self.team_name_var).grid(
            row=2, column=1, columnspan=5, sticky="ew", padx=(6, 0), pady=(8, 0),
        )
        endpoint.grid_columnconfigure(3, weight=1)

        roster_frame = ttk.LabelFrame(self.root, text="Your roster (choose 5)", padding=8)
        roster_frame.pack(fill="both", expand=True, padx=12, pady=6)
        roster_frame.grid_rowconfigure(1, weight=1)
        roster_frame.grid_columnconfigure(0, weight=1)
        roster_frame.grid_columnconfigure(2, weight=1)
        ttk.Label(roster_frame, text="Available characters").grid(row=0, column=0, sticky="w")
        ttk.Label(roster_frame, text="Your selected team").grid(row=0, column=2, sticky="w")
        self.available = tk.Listbox(roster_frame, selectmode="extended", exportselection=False)
        self.available.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.chosen = tk.Listbox(roster_frame, exportselection=False)
        self.chosen.grid(row=1, column=2, sticky="nsew", pady=(4, 0))
        scrollbar = ttk.Scrollbar(roster_frame, orient="vertical", command=self.available.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(4, 0), padx=(3, 8))
        self.available.configure(yscrollcommand=scrollbar.set)
        for name in sorted(all_names(), key=str.casefold):
            self.available.insert("end", name)

        buttons = ttk.Frame(roster_frame)
        buttons.grid(row=1, column=3, padx=(10, 0))
        self.add_button = ttk.Button(buttons, text="Add", command=self.add_selected)
        self.add_button.pack(fill="x", pady=3)
        self.remove_button = ttk.Button(buttons, text="Remove", command=self.remove_selected)
        self.remove_button.pack(fill="x", pady=3)
        self.available.bind("<Double-Button-1>", lambda _event: self.add_selected())
        self.chosen.bind("<Double-Button-1>", lambda _event: self.remove_selected())

        options = ttk.Frame(self.root, padding=(12, 4))
        options.pack(fill="x")
        ttk.Label(options, text="Spike carrier").grid(row=0, column=0, sticky="w")
        self.spike_menu = ttk.OptionMenu(options, self.spike_var, "")
        self.spike_menu.grid(row=0, column=1, sticky="ew", padx=(6, 16))
        ttk.Label(options, text="IGL").grid(row=0, column=2, sticky="w")
        self.igl_menu = ttk.OptionMenu(options, self.igl_var, "")
        self.igl_menu.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        options.grid_columnconfigure(1, weight=1)
        options.grid_columnconfigure(3, weight=1)

        footer = ttk.Frame(self.root, padding=12)
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left", expand=True, anchor="w")
        ttk.Button(footer, text="Connect", command=self.confirm).pack(side="right")

    def select_preset(self, name):
        if name == ORIGINAL_TEAM_OPTION:
            self.preset_active = False
            self.roster.clear()
            self.team_name_var.set("")
            self.spike_var.set("")
            self.igl_var.set("")
            self._refresh_roster()
            return
        preset = get_preset(name)
        if preset is None or preset.validate(all_names(), expected_size=5):
            messagebox.showerror("Invalid team preset", "The selected team preset is not available.", parent=self.root)
            self.preset_var.set(ORIGINAL_TEAM_OPTION)
            return
        self.preset_active = True
        self.roster = list(preset.players)
        self.team_name_var.set(preset.name)
        self.spike_var.set(preset.spike_holder or preset.players[0])
        self.igl_var.set(preset.igl)
        self._refresh_roster()

    def add_selected(self):
        if self.preset_active:
            return
        for index in self.available.curselection():
            name = self.available.get(index)
            if name not in self.roster and len(self.roster) < 5:
                self.roster.append(name)
        self._refresh_roster()

    def remove_selected(self):
        if self.preset_active:
            return
        indexes = set(self.chosen.curselection())
        self.roster = [name for index, name in enumerate(self.roster) if index not in indexes]
        self._refresh_roster()

    def _refresh_roster(self):
        self.chosen.delete(0, "end")
        for name in self.roster:
            self.chosen.insert("end", name)
        state = "disabled" if self.preset_active else "normal"
        self.available.configure(state=state)
        self.add_button.configure(state=state)
        self.remove_button.configure(state=state)
        for variable, menu in ((self.spike_var, self.spike_menu), (self.igl_var, self.igl_menu)):
            menu["menu"].delete(0, "end")
            for name in self.roster:
                menu["menu"].add_command(label=name, command=lambda selected=name, v=variable: v.set(selected))
            if variable.get() not in self.roster:
                variable.set(self.roster[0] if self.roster else "")
        self.status_var.set(f"{len(self.roster)} / 5 characters selected.")

    def confirm(self):
        host = self.host_var.get().strip()
        team = self.team_var.get()
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Invalid port", "Enter a port number from 1 to 65535.", parent=self.root)
            return
        if not host or not 1 <= port <= 65535:
            messagebox.showerror("Invalid connection", "Enter a server IP/host and a port from 1 to 65535.", parent=self.root)
            return
        if len(self.roster) != 5:
            messagebox.showerror("Roster incomplete", "Select a preset or choose exactly five characters.", parent=self.root)
            return
        selection = {
            "roster": list(self.roster),
            "spike_holder_name": self.spike_var.get(),
            "igl_name": self.igl_var.get(),
            "team_name": self.team_name_var.get().strip() or None,
        }
        self.root.destroy()
        self.on_confirm(selection, host, port, team)

    def run(self):
        self.root.mainloop()
