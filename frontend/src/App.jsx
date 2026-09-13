import { useEffect, useState } from "react";

import {
  FiActivity,
  FiAlertTriangle,
  FiBarChart2,
  FiBell,
  FiChevronLeft,
  FiChevronRight,
  FiCpu,
  FiDatabase,
  FiGlobe,
  FiGrid,
  FiLayers,
  FiLink,
  FiMenu,
  FiMessageSquare,
  FiSearch,
  FiServer,
  FiSettings,
  FiShield,
  FiTarget,
  FiUser,
  FiUsers,
  FiX,
  FiZap,
} from "react-icons/fi";

import { getHealthStatus } from "./services/api";
import UrlScanner from "./pages/UrlScanner";


const menuGroups = [
  {
    title: "",
    items: [
      {
        name: "Overview",
        icon: FiGrid,
      },
    ],
  },

  {
    title: "Security",
    items: [
      {
        name: "Threat Dashboard",
        icon: FiActivity,
      },
      {
        name: "URL Scanner",
        icon: FiLink,
      },
      {
        name: "Phishing Scanner",
        icon: FiMessageSquare,
      },
      {
        name: "Network Monitor",
        icon: FiGlobe,
      },
      {
        name: "Behaviour Analytics",
        icon: FiUsers,
      },
    ],
  },

  {
    title: "Intelligence",
    items: [
      {
        name: "Threat Intelligence",
        icon: FiTarget,
      },
      {
        name: "Threat Graph",
        icon: FiLayers,
      },
      {
        name: "Security Events",
        icon: FiDatabase,
      },
    ],
  },

  {
    title: "Response",
    items: [
      {
        name: "Alerts",
        icon: FiBell,
      },
      {
        name: "Incident Center",
        icon: FiAlertTriangle,
      },
    ],
  },

  {
    title: "AI",
    items: [
      {
        name: "AI Security Assistant",
        icon: FiCpu,
      },
      {
        name: "Model Analytics",
        icon: FiBarChart2,
      },
    ],
  },

  {
    title: "System",
    items: [
      {
        name: "Settings",
        icon: FiSettings,
      },
    ],
  },
];


function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [backendData, setBackendData] = useState(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const [activePage, setActivePage] = useState("Overview");


  const checkBackend = async () => {
    try {
      setBackendStatus("checking");

      const response = await getHealthStatus();

      if (
        response.success &&
        response.data?.status === "healthy"
      ) {
        setBackendStatus("online");
        setBackendData(response.data);
      } else {
        setBackendStatus("offline");
        setBackendData(null);
      }
    } catch (error) {
      console.error(
        "Backend connection failed:",
        error
      );

      setBackendStatus("offline");
      setBackendData(null);
    }
  };


  useEffect(() => {
    checkBackend();
  }, []);


  const online =
    backendStatus === "online";


  const handlePageChange = (pageName) => {
    setActivePage(pageName);

    setMobileSidebarOpen(false);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };


  return (
    <div className="min-h-screen bg-[#050b14] text-slate-100">

      {/* Mobile overlay */}

      {mobileSidebarOpen && (
        <button
          onClick={() =>
            setMobileSidebarOpen(false)
          }
          className="fixed inset-0 z-40 bg-black/70 lg:hidden"
          aria-label="Close sidebar"
        />
      )}


      {/* Sidebar */}

      <aside
        className={`
          fixed left-0 top-0 z-50 h-screen
          border-r border-slate-800
          bg-[#07111f]
          transition-all duration-300

          ${
            sidebarCollapsed
              ? "lg:w-20"
              : "lg:w-72"
          }

          ${
            mobileSidebarOpen
              ? "w-72 translate-x-0"
              : "w-72 -translate-x-full lg:translate-x-0"
          }
        `}
      >

        {/* Logo */}

        <div className="flex h-20 items-center border-b border-slate-800 px-5">

          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10">
            <FiShield className="text-2xl text-blue-400" />
          </div>

          {!sidebarCollapsed && (
            <div className="ml-3">

              <h1 className="text-lg font-bold">
                DeepShield AI
              </h1>

              <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                Security Command Center
              </p>

            </div>
          )}


          <button
            onClick={() =>
              setMobileSidebarOpen(false)
            }
            className="ml-auto text-slate-400 lg:hidden"
          >
            <FiX />
          </button>

        </div>


        {/* Navigation */}

        <div className="h-[calc(100vh-130px)] overflow-y-auto px-3 py-5">

          {menuGroups.map((group) => (
            <div
              key={
                group.title ||
                "overview"
              }
              className="mb-5"
            >

              {group.title &&
                !sidebarCollapsed && (

                <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                  {group.title}
                </p>

              )}


              <div className="space-y-1">

                {group.items.map(
                  (item) => {

                    const Icon =
                      item.icon;

                    const isActive =
                      activePage ===
                      item.name;


                    return (
                      <button
                        key={item.name}
                        onClick={() =>
                          handlePageChange(
                            item.name
                          )
                        }
                        title={
                          sidebarCollapsed
                            ? item.name
                            : undefined
                        }
                        className={`
                          flex w-full items-center
                          rounded-xl border
                          px-3 py-2.5
                          text-sm transition

                          ${
                            isActive
                              ? "border-blue-500/20 bg-blue-500/10 text-blue-300"
                              : "border-transparent text-slate-400 hover:bg-slate-900 hover:text-white"
                          }

                          ${
                            sidebarCollapsed
                              ? "justify-center"
                              : "gap-3"
                          }
                        `}
                      >

                        <Icon className="shrink-0 text-lg" />

                        {!sidebarCollapsed && (
                          <span>
                            {item.name}
                          </span>
                        )}

                      </button>
                    );
                  }
                )}

              </div>

            </div>
          ))}

        </div>


        {/* Collapse */}

        <div className="absolute bottom-0 left-0 right-0 border-t border-slate-800 bg-[#07111f] p-3">

          <button
            onClick={() =>
              setSidebarCollapsed(
                !sidebarCollapsed
              )
            }
            className="hidden w-full items-center justify-center rounded-lg p-2 text-slate-500 hover:bg-slate-900 lg:flex"
          >

            {sidebarCollapsed ? (
              <FiChevronRight />
            ) : (
              <>
                <FiChevronLeft />

                <span className="ml-2 text-xs">
                  Collapse Sidebar
                </span>
              </>
            )}

          </button>

        </div>

      </aside>


      {/* Main application */}

      <div
        className={`
          min-h-screen
          transition-all duration-300

          ${
            sidebarCollapsed
              ? "lg:ml-20"
              : "lg:ml-72"
          }
        `}
      >

        {/* Header */}

        <header className="sticky top-0 z-30 flex h-20 items-center border-b border-slate-800 bg-[#050b14]/95 px-4 backdrop-blur md:px-7">

          <button
            onClick={() =>
              setMobileSidebarOpen(true)
            }
            className="mr-3 rounded-lg border border-slate-800 p-2 lg:hidden"
          >
            <FiMenu />
          </button>


          <div className="hidden max-w-xl flex-1 items-center rounded-xl border border-slate-800 bg-slate-950/50 px-4 md:flex">

            <FiSearch className="text-slate-500" />

            <input
              placeholder="Search security events, IP, domain..."
              className="w-full bg-transparent px-3 py-2.5 text-sm outline-none placeholder:text-slate-600"
            />

          </div>


          <div className="ml-auto flex items-center gap-3">

            <button className="rounded-xl border border-slate-800 bg-slate-900 p-2.5 text-slate-400">
              <FiBell />
            </button>


            <div className="hidden items-center gap-3 border-l border-slate-800 pl-4 sm:flex">

              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-300">
                <FiUser />
              </div>

              <div>

                <p className="text-sm font-medium">
                  Security Admin
                </p>

                <p className="text-xs text-slate-600">
                  DeepShield Console
                </p>

              </div>

            </div>

          </div>

        </header>


        {/* Page content */}

        {activePage === "Overview" && (

          <OverviewPage
            backendStatus={
              backendStatus
            }
            backendData={
              backendData
            }
            online={online}
            checkBackend={
              checkBackend
            }
          />

        )}


        {activePage === "URL Scanner" && (
          <UrlScanner />
        )}


        {activePage !== "Overview" &&
          activePage !== "URL Scanner" && (

          <ComingSoonPage
            pageName={activePage}
          />

        )}

      </div>

    </div>
  );
}


function OverviewPage({
  backendStatus,
  backendData,
  online,
  checkBackend,
}) {
  return (
    <main className="p-4 md:p-7">

      {/* Overview heading */}

      <section className="mb-7 flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">

        <div>

          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">
            Security Overview
          </p>

          <h2 className="text-3xl font-semibold tracking-tight text-white">
            AI Security Command Center
          </h2>

          <p className="mt-2 text-sm text-slate-500">
            Monitor AI detection modules,
            security events and system health.
          </p>

        </div>


        <div className="flex w-fit items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">

          <span
            className={`
              h-2.5 w-2.5 rounded-full

              ${
                backendStatus ===
                "online"
                  ? "bg-emerald-400"
                  : backendStatus ===
                      "checking"
                    ? "bg-amber-400"
                    : "bg-red-500"
              }
            `}
          />


          <div>

            <p className="text-[10px] uppercase tracking-wider text-slate-600">
              System Status
            </p>

            <p
              className={
                online
                  ? "text-sm font-semibold text-emerald-400"
                  : backendStatus ===
                      "checking"
                    ? "text-sm font-semibold text-amber-400"
                    : "text-sm font-semibold text-red-400"
              }
            >
              {online
                ? "Protected"
                : backendStatus ===
                    "checking"
                  ? "Checking..."
                  : "Backend Offline"}
            </p>

          </div>

        </div>

      </section>


      {/* Metric cards */}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">

        <MetricCard
          icon={FiShield}
          title="Threats Detected"
          value="0"
          subtitle="No stored events yet"
        />

        <MetricCard
          icon={FiAlertTriangle}
          title="Critical Alerts"
          value="0"
          subtitle="Alert engine pending"
        />

        <MetricCard
          icon={FiZap}
          title="Blocked Attacks"
          value="0"
          subtitle="Protection automation pending"
        />

        <MetricCard
          icon={FiActivity}
          title="Security Score"
          value="—"
          subtitle="Unified risk engine pending"
        />

        <MetricCard
          icon={FiDatabase}
          title="Active Events"
          value="0"
          subtitle="No event source connected"
        />

      </section>


      {/* Activity + modules */}

      <section className="mt-5 grid gap-5 xl:grid-cols-[1.7fr_1fr]">

        <Panel title="Threat Activity">

          <div className="mt-5 flex min-h-[330px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">

            <div className="text-center">

              <FiActivity className="mx-auto text-4xl text-blue-400" />

              <h3 className="mt-4 font-semibold">
                No threat activity yet
              </h3>

              <p className="mt-2 max-w-sm text-sm text-slate-500">
                URL detection is operational.
                Real activity charts will appear
                after security events are stored
                in the database.
              </p>

            </div>

          </div>

        </Panel>


        <Panel title="Detection Modules">

          <div className="mt-5 space-y-3">

            <Module
              name="FastAPI Backend"
              status={
                online
                  ? "Operational"
                  : "Offline"
              }
              ready={online}
            />

            <Module
              name="Malicious URL AI"
              status={
                online
                  ? "Operational"
                  : "Unavailable"
              }
              ready={online}
            />

            <Module
              name="Phishing Message AI"
              status="Planned"
            />

            <Module
              name="Network Intrusion AI"
              status="Planned"
            />

            <Module
              name="Anomaly Detection"
              status="Planned"
            />

            <Module
              name="Unified Risk Engine"
              status="Planned"
            />

          </div>

        </Panel>

      </section>


      {/* Alerts + health */}

      <section className="mt-5 grid gap-5 xl:grid-cols-2">

        <Panel title="Recent Security Alerts">

          <div className="mt-5 rounded-xl border border-slate-800 p-10 text-center">

            <FiBell className="mx-auto text-3xl text-slate-700" />

            <p className="mt-3 text-sm text-slate-500">
              No security alerts detected.
            </p>

          </div>

        </Panel>


        <Panel title="System Health">

          <div className="mt-5 space-y-4">

            <HealthRow
              icon={FiServer}
              name="API Server"
              status={
                online
                  ? "Operational"
                  : "Offline"
              }
              active={online}
            />

            <HealthRow
              icon={FiCpu}
              name="URL AI Engine"
              status={
                online
                  ? "Operational"
                  : "Unavailable"
              }
              active={online}
            />

            <HealthRow
              icon={FiDatabase}
              name="Security Database"
              status="Not Configured"
            />

            <HealthRow
              icon={FiActivity}
              name="Real-Time Monitoring"
              status="Not Configured"
            />

          </div>


          <button
            onClick={checkBackend}
            className="mt-5 rounded-lg border border-slate-800 bg-slate-950 px-4 py-2 text-xs text-slate-400 hover:text-white"
          >
            Refresh Backend Status
          </button>


          {backendData && (

            <p className="mt-4 text-xs text-slate-600">
              Backend v
              {backendData.version}
              {" • "}
              {backendData.environment}
            </p>

          )}

        </Panel>

      </section>


      <footer className="mt-8 border-t border-slate-900 py-5 text-center text-xs text-slate-700">
        DeepShield AI • AI-Powered
        Cybersecurity Platform
      </footer>

    </main>
  );
}


function ComingSoonPage({
  pageName,
}) {
  return (
    <main className="p-4 md:p-7">

      <div className="flex min-h-[calc(100vh-136px)] items-center justify-center">

        <div className="w-full max-w-xl rounded-2xl border border-slate-800 bg-[#091321] p-8 text-center">

          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-blue-500/20 bg-blue-500/10 text-blue-400">

            <FiShield className="text-2xl" />

          </div>


          <p className="mt-6 text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">
            DeepShield AI
          </p>


          <h2 className="mt-2 text-2xl font-semibold text-white">
            {pageName}
          </h2>


          <p className="mt-3 text-sm leading-6 text-slate-500">
            This module is planned for a
            future DeepShield AI phase.
            The URL Security Scanner is
            currently the active AI
            detection module.
          </p>


          <div className="mt-6 inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/50 px-4 py-2 text-xs text-slate-400">

            <FiActivity />

            Development Pending

          </div>

        </div>

      </div>

    </main>
  );
}


function MetricCard({
  icon: Icon,
  title,
  value,
  subtitle,
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#091321] p-5 transition hover:border-slate-700">

      <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-blue-400">

        <Icon className="text-xl" />

      </div>


      <p className="mt-5 text-xs uppercase tracking-wider text-slate-600">
        {title}
      </p>


      <p className="mt-2 text-2xl font-semibold text-white">
        {value}
      </p>


      <p className="mt-1 text-xs text-slate-600">
        {subtitle}
      </p>

    </div>
  );
}


function Panel({
  title,
  children,
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#091321] p-6">

      <h3 className="font-semibold text-slate-100">
        {title}
      </h3>

      {children}

    </div>
  );
}


function Module({
  name,
  status,
  ready = false,
}) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/40 px-4 py-3">

      <div className="flex items-center gap-3">

        <span
          className={`h-2 w-2 rounded-full ${
            ready
              ? "bg-emerald-400"
              : "bg-slate-700"
          }`}
        />

        <p className="text-sm text-slate-300">
          {name}
        </p>

      </div>


      <span
        className={
          ready
            ? "text-xs text-emerald-400"
            : "text-xs text-slate-600"
        }
      >
        {status}
      </span>

    </div>
  );
}


function HealthRow({
  icon: Icon,
  name,
  status,
  active = false,
}) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 pb-4">

      <div className="flex items-center gap-3">

        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-slate-950 text-slate-500">

          <Icon />

        </div>

        <p className="text-sm text-slate-300">
          {name}
        </p>

      </div>


      <span
        className={
          active
            ? "text-xs text-emerald-400"
            : "text-xs text-slate-600"
        }
      >
        {status}
      </span>

    </div>
  );
}


export default App;