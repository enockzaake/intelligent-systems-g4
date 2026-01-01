"use client";

import * as React from "react";
import {
  IconChartBar,
  IconDashboard,
  IconDatabase,
  IconFileAi,
  IconHelp,
  IconInnerShadowTop,
  IconListDetails,
  IconSettings,
  IconUsers,
  IconBrain,
} from "@tabler/icons-react";

import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import Link from "next/link";

const data = {
  user: {
    name: "Fleet Manager",
    email: "admin@fleetai.com",
    avatar: "/avatars/shadcn.jpg",
  },
  navMain: [
    {
      title: "Dashboard",
      url: "/dashboard",
      icon: IconDashboard,
    },
    {
      title: "Route Simulation",
      url: "/dashboard/route-simulation",
      icon: IconChartBar,
    },
    {
      title: "Optimization & Scenarios",
      url: "/dashboard/optimization",
      icon: IconFileAi,
    },
    {
      title: "DL Route Optimizer (v2)",
      url: "/dashboard/dl-optimizer",
      icon: IconBrain,
    },
  ],
  navSecondary: [],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <Link href="/">
                <IconInnerShadowTop className="!size-5" />
                <span className="text-base font-semibold">Fleet AI</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
    </Sidebar>
  );
}
